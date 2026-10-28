// src/poly/dsv4_test.cu - DeepSeek V4's kernels (kernels_dsv4.cu) against the same arithmetic on the host: the rope,
// the compressor, the indexer's scores and pick, and the attention with its window, its compressed rows and sinks.
// It checks the kernels do what their comments say; llama.cpp is the reference for what that should be
// (tools: a model's tokens against llama.cpp's).
//
// usage: poly_dsv4_test   (exit 0: every kernel within float rounding of the host)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

std::mt19937 rng(7);

std::vector<float> random(size_t n, float scale = 1.f) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

std::vector<uint16_t> to_f16(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __half_as_ushort(__float2half(v[i]));
    return h;
}

float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

int report(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs((double) got[i] - want[i]) / (1e-3 + std::abs((double) want[i])));
    const bool ok = worst <= tol;
    std::printf("%-34s %zu values, worst %.2g: %s\n", what, got.size(), worst, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

void rope_host(float* x, int dim, int off, int n_rot, const Rope& r, double pos, bool back) {
    for (int i0 = off; i0 < off + n_rot; i0 += 2) {
        const int iw = i0 - off;
        const float te = (float) pos * std::pow(r.theta_scale, iw / 2.0f);
        float theta = r.freq_scale * te, mscale = r.attn_factor;
        if (r.ext_factor != 0.0f) {
            const float y = (iw / 2 - r.corr0) / std::max(0.001f, r.corr1 - r.corr0);
            const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * r.ext_factor;
            theta = theta * (1 - ramp) + te * ramp;
            mscale *= 1.0f + 0.1f * std::log(1.0f / r.freq_scale);
        }
        const float c = std::cos(theta) * mscale, s = std::sin(theta) * mscale * (back ? -1.f : 1.f);
        const float a = x[i0], b = x[i0 + 1];
        x[i0] = a * c - b * s;
        x[i0 + 1] = a * s + b * c;
    }
}

int test_rope() {
    const int dim = 512, heads = 4, rows = 3, off = 448, n_rot = 64;
    Rope r;
    r.theta_scale = std::pow(160000.f, -2.f / n_rot);
    r.freq_scale = 1.f / 16;
    r.ext_factor = 1.f;
    r.attn_factor = 1.f / (1.f + 0.1f * std::log(16.f));
    r.corr0 = 10, r.corr1 = 40;
    auto x = random((size_t) dim * heads * rows);
    float* d = dev(x);
    dsv4_rope(d, d, dim, heads, off, n_rot, r, 1000, 4, rows, true, 1e-6f, false, nullptr);
    auto got = host(d, x.size());
    auto want = x;
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = want.data() + ((size_t) t * heads + h) * dim;
            double ss = 0;
            for (int i = 0; i < dim; ++i) ss += (double) v[i] * v[i];
            const float sc = 1.f / std::sqrt((float) (ss / dim) + 1e-6f);
            for (int i = 0; i < dim; ++i) v[i] *= sc;
            rope_host(v, dim, off, n_rot, r, 1000 + 4 * t, false);
        }
    int bad = report("rope, normed, YaRN", got, want, 1e-4);
    dsv4_rope(d, d, dim, heads, off, n_rot, r, 1000, 4, rows, false, 0.f, true, nullptr);   // turned back
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) rope_host(want.data() + ((size_t) t * heads + h) * dim, dim, off, n_rot, r,
                                                  1000 + 4 * t, true);
    bad += report("rope back", host(d, x.size()), want, 1e-4);
    cudaFree(d);
    return bad;
}

int test_squeeze(bool overlap, int ratio, int dim) {
    const int width = overlap ? 2 * dim : dim, rows = 3 * ratio + 1, ring = rows + 2 * ratio, n_rot = 64;
    auto kv = random((size_t) rows * width), sc = random((size_t) rows * width), ape = random((size_t) ratio * width);
    auto norm = random(dim, 0.5f);
    std::vector<float> zeros((size_t) ring * width, 0.f);
    float *d_kv = dev(kv), *d_sc = dev(sc), *d_ape = dev(ape), *r_kv = dev(zeros), *r_sc = dev(zeros), *d_n = dev(norm);
    Rope r;
    r.theta_scale = std::pow(160000.f, -2.f / n_rot);
    r.freq_scale = 1.f / 16;
    r.ext_factor = 1.f;
    r.attn_factor = 1.f / (1.f + 0.1f * std::log(16.f));
    r.corr0 = 10, r.corr1 = 40;
    const int64_t pos = 0, count = (pos + rows) / ratio;
    std::vector<uint16_t> zc((size_t) count * dim, 0);
    uint16_t* d_c = dev(zc);
    dsv4_state(d_kv, d_sc, d_ape, r_kv, r_sc, ring, width, ratio, pos, rows, nullptr);
    dsv4_squeeze(r_kv, r_sc, ring, dim, ratio, overlap, d_n, 1e-6f, r, n_rot, pos, rows, d_c, nullptr);
    std::vector<float> got, want;
    for (uint16_t h : host(d_c, zc.size())) got.push_back(f16(h));
    for (int b = 0; b < count; ++b) {
        std::vector<float> rowv(dim);
        for (int d = 0; d < dim; ++d) {
            std::vector<float> s, v;
            const int start = b * ratio;
            for (int j = 0; j < (overlap ? 2 * ratio : ratio); ++j) {
                int row = start + j, col = d;
                if (overlap && j < ratio) row = start - ratio + j;
                else if (overlap) row = start + j - ratio, col = dim + d;
                if (row < 0) continue;
                s.push_back(sc[(size_t) row * width + col] + ape[(size_t) (row % ratio) * width + col]);
                v.push_back(kv[(size_t) row * width + col]);
            }
            const float mx = *std::max_element(s.begin(), s.end());
            float sum = 0, acc = 0;
            for (float x : s) sum += std::exp(x - mx);
            for (size_t j = 0; j < s.size(); ++j) acc += std::exp(s[j] - mx) / sum * v[j];
            rowv[d] = acc;
        }
        double ss = 0;
        for (float x : rowv) ss += (double) x * x;
        const float scale = 1.f / std::sqrt((float) (ss / dim) + 1e-6f);
        for (int d = 0; d < dim; ++d) rowv[d] *= scale * norm[d];
        rope_host(rowv.data(), dim, dim - n_rot, n_rot, r, b * ratio, false);
        for (float x : rowv) want.push_back(f16(__half_as_ushort(__float2half(x))));
    }
    char what[64];
    std::snprintf(what, sizeof what, "squeeze ratio %d%s", ratio, overlap ? ", overlap" : "");
    const int bad = report(what, got, want, 2e-3);
    for (float* p : {d_kv, d_sc, d_ape, r_kv, r_sc, d_n}) cudaFree(p);
    cudaFree(d_c);
    return bad;
}

// rows from GROUP_ROWS (32) take the batch kernel; 12 heads leave its second group of 8 short.  batch 0 (with
// heads a multiple of GROUP_HEADS) takes the tensor-core rows kernel.  q8 quantizes the raw rows as store_q8
// does and gives the reference the dequantized values, so the kernel's unpacking is what the comparison is about
int test_attend(bool with_sel, int ratio, int rows, int heads, int window, int batch = 1, bool q8 = false) {
    const int dim = 512, top = 6;
    const int64_t pos = 60;
    const int ring = window + rows, n_comp = (int) ((pos + rows) / ratio);
    auto q = random((size_t) rows * heads * dim, 0.05f);
    auto rawf = random((size_t) ring * dim), compf = random((size_t) n_comp * dim), sinks = random(heads);
    auto raw = to_f16(rawf), comp = to_f16(compf);
    // q8_0 as store_q8 writes it (kernels.cu): d = amax/127 a 32-value block, codes = rint(x/d) clamped to 127;
    // the reference reads what the cache holds, code * half(d), which is the kernel's mul2 exactly
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    if (q8) {
        codes.assign((size_t) ring * dim, 0);
        scales.assign((size_t) ring * (dim / 32), 0);
        for (int r = 0; r < ring; ++r)
            for (int b = 0; b < dim / 32; ++b) {
                float amax = 0.f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(rawf[(size_t) r * dim + b * 32 + j]));
                const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
                const uint16_t dh = __half_as_ushort(__float2half(d));
                scales[(size_t) r * (dim / 32) + b] = dh;
                for (int j = 0; j < 32; ++j) {
                    const size_t i = (size_t) r * dim + b * 32 + j;
                    const float c = std::min(std::max(std::rint(rawf[i] * id), -127.f), 127.f);
                    codes[i] = (int8_t) c;
                    rawf[i] = c * f16(dh);
                }
            }
        raw = to_f16(rawf);
    }
    std::vector<int32_t> sel((size_t) rows * top);
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < top; ++j) sel[(size_t) t * top + j] = (j * 3 + t) % std::max(1, (int) ((pos + t + 1) / ratio));
    float *d_q = dev(q), *d_s = dev(sinks);
    uint16_t *d_raw = dev(raw), *d_comp = dev(comp);
    int32_t* d_sel = dev(sel);
    int8_t* d_codes = q8 ? dev(codes) : nullptr;
    uint16_t* d_scales = q8 ? dev(scales) : nullptr;
    const Kvq8 kq{d_codes, d_scales};
    std::vector<float> zo((size_t) rows * heads * dim);
    float* d_out = dev(zo);
    dsv4_attend(d_q, d_raw, ring, window, d_comp, ratio, with_sel ? d_sel : nullptr, top, top, d_s, d_out, dim, heads,
                0.04f, pos, rows, nullptr, nullptr, 0, kq, batch, 16, 1, 1);
    auto got = host(d_out, zo.size());
    // the same input once through the classic token-a-block kernel, where batch 0 lands on it: a head count that
    // is not a multiple of GROUP_HEADS (16) takes the fallback, so only those cases get the second run
    const bool classic = heads % 16 != 0;
    std::vector<float> got_rows;
    if (classic) {
        cudaMemset(d_out, 0, zo.size() * sizeof(float));
        dsv4_attend(d_q, d_raw, ring, window, d_comp, ratio, with_sel ? d_sel : nullptr, top, top, d_s, d_out, dim,
                    heads, 0.04f, pos, rows, nullptr, nullptr, 0, kq, 0, 16, 1, 1);
        got_rows = host(d_out, zo.size());
    }
    std::vector<float> want(zo.size());
    for (int t = 0; t < rows; ++t) {
        const int64_t p = pos + t;
        std::vector<const uint16_t*> keys;
        for (int64_t k = std::max<int64_t>(0, p - window + 1); k <= p; ++k) keys.push_back(raw.data() + (k % ring) * dim);
        const int vis = (int) ((p + 1) / ratio);
        if (with_sel)
            for (int j = 0; j < std::min(vis, top); ++j) keys.push_back(comp.data() + (size_t) sel[(size_t) t * top + j] * dim);
        else
            for (int b = 0; b < vis; ++b) keys.push_back(comp.data() + (size_t) b * dim);
        for (int h = 0; h < heads; ++h) {
            const float* qh = q.data() + ((size_t) t * heads + h) * dim;
            std::vector<float> sc(keys.size());
            float mx = sinks[h];
            for (size_t k = 0; k < keys.size(); ++k) {
                float d = 0;
                for (int i = 0; i < dim; ++i) d += qh[i] * f16(keys[k][i]);
                sc[k] = d * 0.04f;
                mx = std::max(mx, sc[k]);
            }
            float sum = std::exp(sinks[h] - mx);
            for (float& s : sc) sum += s = std::exp(s - mx);
            for (int i = 0; i < dim; ++i) {
                float acc = 0;
                for (size_t k = 0; k < keys.size(); ++k) acc += sc[k] * f16(keys[k][i]);
                want[((size_t) t * heads + h) * dim + i] = acc / sum;
            }
        }
    }
    // the first places the engine and the reference differ, for a kernel being chased: t's parity tells the
    // pair's half, h a head band, i a value band
    const auto first_diffs = [&](const char* tag, const std::vector<float>& g) {
        int shown = 0;
        for (int t = 0; t < rows && shown < 6; ++t)
            for (int h = 0; h < heads && shown < 6; ++h)
                for (int i = 0; i < dim && shown < 6; ++i) {
                    const size_t k = ((size_t) t * heads + h) * dim + (size_t) i;
                    if (std::abs((double) g[k] - want[k]) > 2e-3 * (1.0 + std::abs((double) want[k]))) {
                        std::printf("  %s diff: t=%d h=%d i=%d got=%+.4f want=%+.4f\n", tag, t, h, i, g[k], want[k]);
                        ++shown;
                    }
                }
    };
    first_diffs(batch ? "pair" : "mma", got);
    if (classic) first_diffs("rows", got_rows);
    {   // the worst place, for a kernel being chased: its t/h/i says the error's geometry
        int wt = 0, wh = 0, wi = 0;
        double worst = 0;
        for (int t = 0; t < rows; ++t)
            for (int h = 0; h < heads; ++h)
                for (int i = 0; i < dim; ++i) {
                    const size_t k = ((size_t) t * heads + h) * dim + (size_t) i;
                    const double e = std::abs((double) got[k] - want[k]);
                    if (e > worst) { worst = e; wt = t; wh = h; wi = i; }
                }
        if (worst > 2e-3)
            std::printf("  %s worst: t=%d h=%d i=%d got=%+.4f want=%+.4f\n", batch ? "pair" : "mma",
                        wt, wh, wi, got[((size_t) wt * heads + wh) * dim + (size_t) wi],
                        want[((size_t) wt * heads + wh) * dim + (size_t) wi]);
    }
    char what[64];
    int bad;
    if (batch) {
        std::snprintf(what, sizeof what, "attend ratio %d%s, %d tokens%s", ratio, with_sel ? ", picked" : "", rows,
                      q8 ? ", q8" : "");
        bad = report(what, got, want, 1e-3);
    } else {
        // the mma path keeps the probabilities and the queries as halves (fdec's class): outputs that cancel to
        // near zero carry an absolute error of up to ~1e-3 no per-value relative bound can hold, so it is judged
        // by the absolute error alone
        std::snprintf(what, sizeof what, "attend mma ratio %d%s, %d tokens%s", ratio, with_sel ? ", picked" : "", rows,
                      q8 ? ", q8" : "");
        double worst_abs = 0;
        for (size_t k = 0; k < zo.size(); ++k) worst_abs = std::max(worst_abs, std::abs((double) got[k] - want[k]));
        std::printf("%-34s %d values, worst abs %.2g: %s\n", what, (int) zo.size(), worst_abs,
                    worst_abs <= 3e-3 ? "ok" : "FAIL");
        bad = worst_abs <= 3e-3 ? 0 : 1;
    }
    if (classic) {
        std::snprintf(what, sizeof what, "attend rows ratio %d%s, %d tokens%s", ratio, with_sel ? ", picked" : "", rows,
                      q8 ? ", q8" : "");
        bad += report(what, got_rows, want, 1e-3);
    }
    for (void* p : {(void*) d_q, (void*) d_s, (void*) d_raw, (void*) d_comp, (void*) d_sel, (void*) d_out,
                    (void*) d_codes, (void*) d_scales}) cudaFree(p);
    return bad;
}

int test_index() {
    const int heads = 64, dim = 128, ratio = 4, top = 16, rows = 3;
    const int64_t pos = 400;
    const int stride = (int) ((pos + rows) / ratio) + 1, n_keys = stride;
    auto q = random((size_t) rows * heads * dim, 0.1f), w = random((size_t) rows * heads);
    auto keysf = random((size_t) n_keys * dim);
    auto keys = to_f16(keysf);
    float *d_q = dev(q), *d_w = dev(w);
    uint16_t* d_k = dev(keys);
    std::vector<float> zs((size_t) rows * stride, -1e30f);
    float* d_s = dev(zs);
    std::vector<int32_t> zsel((size_t) rows * top);
    int32_t* d_sel = dev(zsel);
    dsv4_idx_score(d_q, d_w, d_k, d_s, stride, dim, heads, ratio, pos, rows, 0.01f, nullptr, 0, 0, 0);
    dsv4_select(d_s, stride, d_sel, top, top, ratio, pos, rows, nullptr);
    auto got = host(d_s, zs.size());
    auto sel = host(d_sel, zsel.size());
    // the same scores once through the tensor-core kernel: its queries are halves (fdec's class), so its sums
    // carry a relative error of up to ~1e-2 where the float reference cancels, and it is judged at 2e-2
    dsv4_idx_score(d_q, d_w, d_k, d_s, stride, dim, heads, ratio, pos, rows, 0.01f, nullptr, 0, 1);
    auto got_mma = host(d_s, zs.size());
    std::vector<float> g, want;
    int bad = 0;
    for (int t = 0; t < rows; ++t) {
        const int vis = (int) ((pos + t + 1) / ratio);
        std::vector<float> sc(vis);
        for (int b = 0; b < vis; ++b) {
            float acc = 0;
            for (int h = 0; h < heads; ++h) {
                float d = 0;
                for (int i = 0; i < dim; ++i) d += q[((size_t) t * heads + h) * dim + i] * f16(keys[(size_t) b * dim + i]);
                acc += std::max(d, 0.f) * w[(size_t) t * heads + h];
            }
            sc[b] = acc * 0.01f;
            g.push_back(got[(size_t) t * stride + b]);
            want.push_back(sc[b]);
        }
        // the pick: the top scores the kernel computed, in the order of the rows
        std::vector<int> order(vis);
        std::iota(order.begin(), order.end(), 0);
        const float* mine = got.data() + (size_t) t * stride;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return mine[a] > mine[b]; });
        order.resize(top);
        std::sort(order.begin(), order.end());
        for (int j = 0; j < top; ++j) bad += sel[(size_t) t * top + j] != order[j];
    }
    bad = bad ? 1 : 0;
    std::printf("%-34s %d tokens: %s\n", "indexer pick", rows, bad ? "FAIL" : "ok");
    bad += report("indexer scores", g, want, 1e-3);
    std::vector<float> gm;
    for (int t = 0; t < rows; ++t)
        for (int b = 0; b < (int) ((pos + t + 1) / ratio); ++b)
            gm.push_back(got_mma[(size_t) t * stride + b]);
    // its queries are halves (fdec's class): ~1e-2 where the float reference cancels, judged at 2e-2
    bad += report("indexer scores mma", gm, want, 2e-2);
    for (void* p : {(void*) d_q, (void*) d_w, (void*) d_k, (void*) d_s, (void*) d_sel}) cudaFree(p);
    return bad;
}

// hc_pre's and hc_head's mix, both shapes, against the host's dots
int test_hc() {
    const int hc = 4, n_embd = 512, rows = 40;
    auto r = random((size_t) rows * hc * n_embd, 0.1f);
    int bad = 0;
    for (int n_mix : {hc * (2 + hc), hc}) {
        auto fn = random((size_t) n_mix * hc * n_embd, 0.1f);
        std::vector<float> mix((size_t) rows * (n_mix + 1), -1.f);
        float *d_r = dev(r), *d_fn = dev(fn), *d_m = dev(mix);
        hc_mix(d_r, d_fn, d_m, n_embd, rows, n_mix, nullptr);
        auto got = host(d_m, mix.size());
        std::vector<float> want(mix.size());
        for (int t = 0; t < rows; ++t) {
            const float* rt = r.data() + (size_t) t * hc * n_embd;
            for (int o = 0; o <= n_mix; ++o) {
                float acc = 0;
                if (o < n_mix)
                    for (int i = 0; i < hc * n_embd; ++i) acc += fn[(size_t) o * hc * n_embd + i] * rt[i];
                else
                    for (int i = 0; i < hc * n_embd; ++i) acc += rt[i] * rt[i];
                want[(size_t) t * (n_mix + 1) + o] = acc;
            }
        }
        char what[64];
        std::snprintf(what, sizeof what, "hc mix n_mix %d", n_mix);
        bad += report(what, got, want, 1e-3);
        for (void* p : {(void*) d_r, (void*) d_fn, (void*) d_m}) cudaFree(p);
    }
    return bad;
}

}  // namespace

int main() {
    int bad = test_rope();
    bad += test_squeeze(true, 4, 512);
    bad += test_squeeze(true, 4, 128);
    bad += test_squeeze(false, 128, 512);
    bad += test_attend(false, 128, 5, 8, 16);
    bad += test_attend(true, 4, 5, 8, 16);
    bad += test_attend(false, 4, 5, 8, 16);
    bad += test_attend(false, 128, 40, 12, 40);
    bad += test_attend(true, 4, 40, 12, 40);
    bad += test_attend(false, 4, 40, 12, 40);
    bad += test_attend(true, 4, 41, 12, 40);   // an odd batch: its last block attends one token, its pair's second
                                               // past the batch's end
    bad += test_attend(true, 4, 40, 16, 40, 0);    // the tensor-core rows kernel: the heads on the mma's M tile
    bad += test_attend(false, 128, 40, 32, 40, 0);
    // q8_0 with a window past 64 keys: a second phase, and a P V step that straddles the raw window.  The three
    // bugs of commit 91bd723 (the straddling step's unpacking, the phase's key mask, the 8x mma of the q8 fast
    // path) each showed here and nowhere in the runs above
    bad += test_attend(true, 4, 40, 16, 70, 0, true);
    bad += test_attend(false, 4, 40, 16, 70, 0, true);
    bad += test_attend(true, 128, 40, 16, 70, 0, true);   // ratio 128: the window is the token's whole key list
    bad += test_index();
    bad += test_hc();
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("CUDA error\n");
        return 2;
    }
    return bad ? 1 : 0;
}
