# EE 511 Homework 1 — Problem 1

**Inner Products, Norms, and Angular Similarity** (25 pts)

This file is the working write-up for Part A, Problem 1. Vectors are columns in \(\mathbb{R}^n\). \(\|\cdot\|_p\) is the \(\ell_p\) norm. One MAC is one multiply–accumulate.

The whole problem is one story. An inner product \(x^\top y\) grows if you stretch either vector, so it is not a pure measure of angle. Cosine similarity removes the lengths. Cauchy–Schwarz is the inequality that makes that cosine a real number in \([-1,1]\). Once every vector is forced onto the unit sphere offline, cosine collapses to a bare dot product, which is what a MAC array wants. The last part asks which norm should set an INT8 scale: the representable set is a box, so the right radius is \(\|x\|_\infty\).

---

## (a) Cauchy–Schwarz

**Claim.** For all \(x,y\in\mathbb{R}^n\),

\[
\bigl|x^\top y\bigr| \;\le\; \|x\|_2\,\|y\|_2.
\]

Equality holds if and only if \(x\) and \(y\) are linearly dependent: there exists \(\lambda\in\mathbb{R}\) such that \(x=\lambda y\) or \(y=\lambda x\). (This includes either vector being zero.)

### Proof

**Case \(y=0\).** Then \(x^\top y=0\) and \(\|y\|_2=0\), so both sides are \(0\). Equality holds, and \(y=0\cdot x\), so the two vectors are linearly dependent. The case \(x=0\) is identical.

**Case \(y\neq 0\).** For every \(t\in\mathbb{R}\) the squared residual is nonnegative:

\[
g(t) \;=\; \|x-ty\|_2^2 \;\ge\; 0.
\]

Expand the inner product:

\[
g(t)
\;=\; (x-ty)^\top(x-ty)
\;=\; \|y\|_2^2\, t^2 \;-\; 2(x^\top y)\, t \;+\; \|x\|_2^2.
\]

This is a quadratic \(at^2+bt+c\) with leading coefficient \(a=\|y\|_2^2>0\). A parabola that opens upward and never goes below the axis cannot have two distinct real roots, so its discriminant satisfies \(\Delta\le 0\):

\[
\Delta
\;=\; 4(x^\top y)^2 \;-\; 4\|y\|_2^2\|x\|_2^2
\;\le\; 0.
\]

Cancel \(4\) and take nonnegative square roots:

\[
\bigl|x^\top y\bigr| \;\le\; \|x\|_2\,\|y\|_2.
\]

### Equality, both directions

\(\Rightarrow\). Suppose equality holds. If \(y=0\) we are already done. If \(y\neq 0\), then \(\Delta=0\), so \(g\) has a double root \(t_\star\). Then \(g(t_\star)=0\), hence \(x-t_\star y=0\), hence \(x=t_\star y\).

\(\Leftarrow\). Suppose \(x=\lambda y\). Then

\[
\bigl|x^\top y\bigr| \;=\; |\lambda|\,\|y\|_2^2,
\qquad
\|x\|_2\|y\|_2 \;=\; |\lambda|\,\|y\|_2^2.
\]

The two sides match. If \(\lambda\ge 0\) the inner product is \(+\|x\|_2\|y\|_2\); if \(\lambda<0\) it is \(-\|x\|_2\|y\|_2\). The absolute value saturates in both geometries.

Geometrically: \(g(t)\) is the squared length of \(x\) after subtracting its component along \(y\). That residual is zero only when \(x\) already lives on the line through \(y\). Cauchy–Schwarz is the statement that a projection cannot be longer than the vector itself.

---

## (b) Cosine similarity is well defined

Let \(x,y\neq 0\). The denominator \(\|x\|_2\|y\|_2\) is then strictly positive, so we may divide the inequality of part (a) by it:

\[
\left|\frac{x^\top y}{\|x\|_2\|y\|_2}\right| \;\le\; 1
\qquad\Rightarrow\qquad
\cos\theta \;\in\; [-1,1].
\]

Without Cauchy–Schwarz the same formula could return a number outside \([-1,1]\), which cannot be the cosine of an angle. That is what “well defined” means here.

**Pair with \(\cos\theta=+1\).** Take \(x=(1,0,0)^\top\) and \(y=(2,0,0)^\top\) in \(\mathbb{R}^3\). Then \(x^\top y=2\) and \(\|x\|_2\|y\|_2=2\), so \(\cos\theta=+1\). The two vectors are parallel and point in the same direction (angle \(0\)).

**Pair with \(\cos\theta=-1\).** Take \(x=(1,0,0)^\top\) and \(y=(-3,0,0)^\top\). Then \(x^\top y=-3\) and \(\|x\|_2\|y\|_2=3\), so \(\cos\theta=-1\). The two vectors are parallel and point in opposite directions (angle \(\pi\)).

---

## (c) Norm equivalence

We prove, for every \(x\in\mathbb{R}^n\),

\[
\|x\|_2 \;\le\; \|x\|_1
\qquad\text{and}\qquad
\|x\|_1 \;\le\; \sqrt{n}\,\|x\|_2.
\]

### First inequality: \(\|x\|_2\le\|x\|_1\)

Compare squares. Expanding the \(\ell_1\) square produces the \(\ell_2\) square plus a pile of nonnegative cross terms:

\[
\|x\|_1^2
\;=\; \Bigl(\sum_{i=1}^n |x_i|\Bigr)^2
\;=\; \sum_{i=1}^n x_i^2 \;+\; \sum_{i\neq j}|x_i||x_j|
\;\ge\; \sum_{i=1}^n x_i^2
\;=\; \|x\|_2^2.
\]

Both norms are nonnegative, so \(\|x\|_2\le\|x\|_1\).

**Equality.** Every cross term vanishes if and only if at most one coordinate is nonzero. An extremal vector is therefore **sparse**: any 1-sparse vector, e.g. \(x=e_1=(1,0,\ldots,0)^\top\). For this \(x\) both norms equal \(1\).

### Second inequality: \(\|x\|_1\le\sqrt{n}\,\|x\|_2\)

Apply part (a) to the absolute-value vector \(z=(|x_1|,\ldots,|x_n|)^\top\) and the all-ones vector \(\mathbf{1}=(1,\ldots,1)^\top\):

\[
z^\top\mathbf{1} \;=\; \|x\|_1,
\qquad
\|z\|_2 \;=\; \|x\|_2,
\qquad
\|\mathbf{1}\|_2 \;=\; \sqrt{n}.
\]

Cauchy–Schwarz gives \(\|x\|_1\le\sqrt{n}\,\|x\|_2\).

**Equality.** \(z\) and \(\mathbf{1}\) are linearly dependent if and only if all \(|x_i|\) are equal. An extremal vector is therefore **dense**: e.g. \(x=(1,1,\ldots,1)^\top\), or any equal-magnitude sign pattern. For the all-ones vector, \(\|x\|_1=n\) and \(\sqrt{n}\|x\|_2=n\).

The two bounds are saturated at opposite extremes: \(\ell_2\le\ell_1\) on sparse vectors, \(\ell_1\le\sqrt{n}\,\ell_2\) on dense equal-magnitude vectors. At embedding dimension \(d=768\) the factor \(\sqrt{n}\approx 27.7\), so the two norms can differ by more than an order of magnitude on a dense activation.

---

## Why pre-normalization is legal

If \(\|x\|_2=\|y\|_2=1\), then

\[
\|x-y\|_2^2 \;=\; 2-2x^\top y,
\]

so ranking by Euclidean distance, by cosine, and by inner product are the same ranking. A vector database may store unit embeddings and implement nearest-neighbor search as a pure MAC-array sweep. Part (d) prices the alternative.

---

## (d) Why accelerators pre-normalize

**Machine.** \(N=10^6\) stored embeddings, dimension \(d=768\), FP32. A 768-lane MAC array retires one full 768-element dot product per cycle. Shared scalar units are not pipelined: one square root costs 20 cycles, one division costs 30 cycles, and the MAC array stalls for the entire scalar latency. The query’s own norm is computed once and cached. The \(\ell_2\) norm of a stored vector is itself a self-dot product, then a square root.

### (i) Raw cosine

| Operation | Count | Reason |
|---|---|---|
| Dot products | \(2N+1=2{,}000{,}001\) | \(q^\top q\) once; then \(v_i^\top v_i\) and \(q^\top v_i\) for each stored vector |
| Square roots | \(N+1=1{,}000{,}001\) | \(\|q\|_2\) once and \(\|v_i\|_2\) for each stored vector |
| Divisions | \(N=1{,}000{,}000\) | one \(\dfrac{q^\top v_i}{\|q\|_2\|v_i\|_2}\) per stored vector |

The product of the two norms is a multiply, not a divide, so it is not counted above.

### (ii) Pre-normalized to unit \(\ell_2\)

Every stored vector and the query are already unit length before they reach the accelerator. Cosine **is** the dot product. The scalar units are used zero times.

| Operation | Count |
|---|---|
| Dot products | \(N=1{,}000{,}000\) |
| Square roots | \(0\) |
| Divisions | \(0\) |

### (iii) Cycles and speedup on the 768-lane array

\[
C_{\mathrm{raw}}
\;=\; (2N+1)\cdot 1 \;+\; (N+1)\cdot 20 \;+\; N\cdot 30
\;=\; 52N+21
\;=\; 52{,}000{,}021
\]

\[
C_{\mathrm{pre}}
\;=\; N\cdot 1
\;=\; 1{,}000{,}000
\]

\[
\mathrm{speedup}
\;=\; \frac{C_{\mathrm{raw}}}{C_{\mathrm{pre}}}
\;=\; \frac{52{,}000{,}021}{1{,}000{,}000}
\;=\; 52.000021
\;\approx\; 52\times
\]

This is a ratio of **total** cycles. A ratio of scalar-unit cycles alone is undefined in case (ii), because that case never touches the scalar unit.

**Where did the removed work go?** It was relocated, not eliminated. Each stored vector pays one self-dot, one square root, and a rescale **once at ingest**, when the unit vector is written into the database. The query is normalized once on the host before it is issued. That offline / one-time cost is amortized across every future query. The accelerator’s critical path keeps only the \(N\) inner products.

### (iv) Same optimization, 1-lane array

A dot product now costs \(768\) cycles.

\[
C_{\mathrm{raw}}
\;=\; (2N+1)\cdot 768 \;+\; (N+1)\cdot 20 \;+\; N\cdot 30
\;=\; 1586N+788
\;=\; 1{,}586{,}000{,}788
\]

\[
C_{\mathrm{pre}}
\;=\; N\cdot 768
\;=\; 768{,}000{,}000
\]

\[
\mathrm{speedup}
\;=\; \frac{1{,}586{,}000{,}788}{768{,}000{,}000}
\;\approx\; 2.065\times
\]

On the wide array a full inner product is 1 cycle and a stalled sqrt/div is 20–30 cycles, so the scalar tax dominates (\(\approx 96\%\) of raw runtime). On a 1-lane machine the two extra dots already cost \(1536N\) cycles and the scalar work is only \(50N\) cycles (\(\approx 3\%\)). The same algebraic rewrite looks unimpressive because you are already paying hundreds of cycles inside the inner product. Scalar-unit latency is worth engineering around only when the vector unit has already made a full dot product cheap.

---

## (e) Choosing a norm for the quantizer

Symmetric INT8 quantization represents integers in \(\{-127,\ldots,127\}\). Dequantization is \(\hat x_i=q_i\cdot s\), so the representable set is the axis-aligned box \([-127s,127s]^n\). A tensor \(x\) fits in that box with no clipping if and only if \(\|x\|_\infty\le 127s\). The finest such scale is therefore

\[
s \;=\; \frac{\|x\|_\infty}{127}.
\]

\(\|x\|_2\) and \(\|x\|_1\) are the wrong shapes. They describe a Euclidean ball and an \(\ell_1\) diamond. INT8’s codebook is a cube. Using \(\|x\|_2/127\) would still avoid clipping (because \(\|x\|_2\ge\|x\|_\infty\)), but the step \(s\) would be coarser than necessary and codes would be wasted. \(\|x\|_1\) is even coarser on a dense tensor.

**Hardware contrast, \(\ell_\infty\) vs \(\ell_2\).** Computing \(\|x\|_\infty\) is a max-reduction tree: only comparisons, depth \(\lceil\log_2 n\rceil\), fixed data-independent latency, no multiplier and no square root. Computing \(\|x\|_2\) needs \(n\) squares, an adder-reduction tree, and a square-root unit. The sqrt is a long-latency iterative datapath. Even when the iteration count is hard-wired, the latency is larger and the pipe is a real floating-point unit rather than a compare tree. A quantizer frontend wants the max.

**When an \(\ell_\infty\) scale is a bad choice.** Transformer activations often contain a single huge outlier. The scale is then set by that one coordinate, and the remaining \(n-1\) values all collapse into a few INT8 bins around zero. Of the 256 available codes, only a handful are used by the mass of the tensor.

**Practical fixes.**

1. Change *which* value sets the scale: clip, or take a high percentile instead of the true max, so typical values get a finer grid.
2. Change *how many* values share one scale: per-channel, per-token, or group quantization, so one outlier pollutes only its group.

---

## Checkpoint

- (a)–(c) are the proofs: quantifiers, the zero vector, both directions of equality, sparse vs dense extremals.
- (d)–(e) are the engineering: \(52{,}000{,}021\) vs \(1{,}000{,}000\) cycles (\(52.000021\times\)); \(1{,}586{,}000{,}788\) vs \(768{,}000{,}000\) cycles (\(\approx 2.065\times\)); scale \(s=\|x\|_\infty/127\).

Next up in this file: Problem 2 (spectra, curvature, step-size ceiling), when we start it.
