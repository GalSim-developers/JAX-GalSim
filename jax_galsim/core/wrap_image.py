# from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def wrap_nonhermition(im, xmin, ymin, nxwrap, nywrap):
    def _body_j(j, vals):
        i, im = vals

        ii = (i - ymin) % nywrap + ymin
        jj = (j - xmin) % nxwrap + xmin

        im = jax.lax.cond(
            # weird way to say if ii != i and jj != j
            # I tried other ways and got test failures
            jnp.abs(ii - i) + jnp.abs(jj - j) != 0,
            lambda im, i, j, ii, jj: im.at[ii, jj].add(im[i, j]),
            lambda im, i, j, ii, jj: im,
            im, i, j, ii, jj,
        )

        return [i, im]

    def _body_i(i, vals):
        im = vals
        _, im = jax.lax.fori_loop(0, im.shape[1], _body_j, [i, im])
        return im

    im = jax.lax.fori_loop(0, im.shape[0], _body_i, im)
    return im


# # the code in this file follows closely the C++ code in galsim/core/Image.cpp
# # for wrapping images onto subimages


# # from this C++ function:
# # // Add row j to row jj
# # // ptr and ptrwrap should be at the start of the respective rows.
# # // At the end of this function, each will be one past the end of the row.
# # template <typename T>
# # void wrap_row(T*& ptr, T*& ptrwrap, int m, int step)
# # {
# #     // Add contents of row j to row jj
# #     if (step == 1)
# #         for(; m; --m) *ptrwrap++ += *ptr++;
# #     else
# #         for(; m; --m,ptr+=step,ptrwrap+=step) *ptrwrap += *ptr;
# # }
# def _wrap_row(im, ptr, ptrwrap, m, step):
#     """Add row `ptr` to row `ptrwrap` in an image `im`
#     with # of columns `m` and at step sizes `step`"""
#     return im.at[ptrwrap, :m:step].add(im.at[ptr, :m:step])


# """
# template <typename T>
# void wrap_cols(T*& ptr, int m, int mwrap, int i1, int i2, int step)
# {
#     int ii = i2 - (i2 % mwrap);
#     if (ii == i2) ii = i1;
#     T* ptrwrap = ptr + ii*step;
#     // First do i in [0,i1).
#     for(int i=0; i<i1;) {
#         xdbg<<"Start loop at i = "<<i<<std::endl;
#         // How many do we do before looping back
#         int k = i2-ii;
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) *ptrwrap++ += *ptr++;
#         else
#             for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
#         ii = i1;
#         ptrwrap -= mwrap*step;
#     }
#     // Skip ahead to do i in [i2,m)
#     assert(ii == i1);
#     assert(ptr == ptrwrap);
#     ptr += mwrap * step;
#     for(int i=i2; i<m;) {
#         xdbg<<"Start loop at i = "<<i<<std::endl;
#         // How many do we do before looping back or ending.
#         int k = std::min(m-i, mwrap);
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) *ptrwrap++ += *ptr++;
#         else
#             for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
#         ptrwrap -= mwrap*step;
#     }
# }
# """
# def wrap_cols(im, ptr, m, mwrap, i1, i2, step):
#     """Wrap row `ptr` into columns [i1,i2) in an image `im`
#     with # of columns `m`, # of columns for wrapping `mwrap`,
#     and at step sizes `step`."""
#     # from this code:
#     # int ii = i2 - (i2 % mwrap);
#     # if (ii == i2) ii = i1;
#     # T* ptrwrap = ptr + ii*step;
#     # note the ptr is actually a row index
#     ii = i2 - (i2 % mwrap)
#     ii = jax.lax.cond(
#         ii == i2,
#         lambda ii, i1: i1,
#         lambda ii, i2: ii,
#         ii,
#         i1,
#     )
#     # this is the starting column index
#     # ptrwrap = ii * step

#     # from this code:
#     # // First do i in [0,i1).
#     # for(int i=0; i<i1;) {
#     #     xdbg<<"Start loop at i = "<<i<<std::endl;
#     #     // How many do we do before looping back
#     #     int k = i2-ii;
#     #     xdbg<<"k = "<<k<<std::endl;
#     #     if (step == 1)
#     #         for (; k; --k, ++i) *ptrwrap++ += *ptr++;
#     #     else
#     #         for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
#     #     ii = i1;
#     #     ptrwrap -= mwrap*step;
#     # }
#     # def _body_0_i1(i, vals):
#     raise NotImplementedError("wrap_cols not implemented yet")


# # from this C++ function
# # // Add conjugate of row j to row jj
# # // ptrwrap should be at the end of the conjugate row, not the beginning.
# # // At the end of this function, ptr will be one past the end of the row, and ptrskip will be
# # // one before the beginning.
# # template <typename T>
# # void wrap_row_conj(T*& ptr, T*& ptrwrap, int m, int step)
# # {
# #     if (step == 1)
# #         for(; m; --m) *ptrwrap-- += CONJ(*ptr++);
# #     else
# #         for(; m; --m,ptr+=step,ptrwrap-=step) *ptrwrap += CONJ(*ptr);
# # }
# def wrap_row_conj(im, ptr, ptrwrap, m, step):
#     """Add conjugate of row `ptr` but reversed to row `ptrwrap` in an image `im`
#     with # of columns `m` and at step sizes `step`."""
#     return im.at[ptrwrap, :m:step].add(im.at[ptr, :m:step][::-1].conj())


# """
# // If j == jj, this needs to be slightly different.
# template <typename T>
# void wrap_row_selfconj(T*& ptr, T*& ptrwrap, int m, int step)
# {
#     if (step == 1)
#         for(int i=(m+1)/2; i; --i,++ptr,--ptrwrap) {
#             *ptrwrap += CONJ(*ptr);
#             *ptr = CONJ(*ptrwrap);
#         }
#     else
#         for(int i=(m+1)/2; i; --i,ptr+=step,ptrwrap-=step) {
#             *ptrwrap += CONJ(*ptr);
#             *ptr = CONJ(*ptrwrap);
#         }
#     ptr += (m-(m+1)/2) * step;
#     ptrwrap -= (m-(m+1)/2) * step;
# }

# // Wrap two half-rows where one has the conjugate information for the other.
# // ptr1 and ptr2 should start at the the pointer for i=mwrap within the two rows.
# // At the end of this function, they will each be one past the end of the rows.
# template <typename T>
# void wrap_hermx_cols_pair(T*& ptr1, T*& ptr2, int m, int mwrap, int step)
# {
#     // We start the wrapping with col N/2 (aka i2-1), which needs to wrap its conjugate
#     // (-N/2) onto itself.
#     // Then as i increases, we decrease ii and continue wrapping conjugates.
#     // When we get to something that wraps onto col 0 (the first one will correspond to
#     // i=-N, which is the conjugate of what is stored at i=N in the other row), we need to
#     // repeat with a regular non-conjugate wrapping for the positive col (e.g. i=N,j=j itself)
#     // which also wraps onto col 0.
#     // Then we run ii back up wrapping normally, until we get to N/2 again (aka i2-1).
#     // The negative col will wrap normally onto -N/2, which means we need to also do a
#     // conjugate wrapping onto N/2.

#     xdbg<<"Start hermx_cols_pair\n";
#     T* ptr1wrap = ptr1;
#     T* ptr2wrap = ptr2;
#     int i = mwrap-1;
#     while (1) {
#         xdbg<<"Start loop at i = "<<i<<std::endl;
#         // Do the first column with a temporary to avoid overwriting.
#         T temp = *ptr1;
#         *ptr1wrap += CONJ(*ptr2);
#         *ptr2wrap += CONJ(temp);
#         ptr1 += step;
#         ptr2 += step;
#         ptr1wrap -= step;
#         ptr2wrap -= step;
#         ++i;
#         // Progress as normal (starting at i=mwrap for the first loop).
#         int k = std::min(m-i, mwrap-2);
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) {
#                 *ptr1wrap-- += CONJ(*ptr2++);
#                 *ptr2wrap-- += CONJ(*ptr1++);
#             }
#         else
#             for (; k; --k, ++i, ptr1+=step, ptr2+=step, ptr1wrap-=step, ptr2wrap-=step) {
#                 *ptr1wrap += CONJ(*ptr2);
#                 *ptr2wrap += CONJ(*ptr1);
#             }
#         xdbg<<"i = "<<i<<std::endl;
#         if (i == m) break;
#         // On the last one, don't increment ptrs, since we need to repeat with the non-conj add.
#         *ptr1wrap += CONJ(*ptr2);
#         *ptr2wrap += CONJ(*ptr1);
#         k = std::min(m-i, mwrap-1);
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) {
#                 *ptr1wrap++ += *ptr1++;
#                 *ptr2wrap++ += *ptr2++;
#             }
#         else
#             for (; k; --k, ++i, ptr1+=step, ptr2+=step, ptr1wrap+=step, ptr2wrap+=step) {
#                 *ptr1wrap += *ptr1;
#                 *ptr2wrap += *ptr2;
#             }
#         xdbg<<"i = "<<i<<std::endl;
#         if (i == m) break;
#         *ptr1wrap += *ptr1;
#         *ptr2wrap += *ptr2;
#     }
# }

# // Wrap a single half-row that is its own conjugate (i.e. j==0)
# template <typename T>
# void wrap_hermx_cols(T*& ptr, int m, int mwrap, int step)
# {
#     xdbg<<"Start hermx_cols\n";
#     T* ptrwrap = ptr;
#     int i = mwrap-1;
#     while (1) {
#         xdbg<<"Start loop at i = "<<i<<std::endl;
#         int k = std::min(m-i, mwrap-1);
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) *ptrwrap-- += CONJ(*ptr++);
#         else
#             for (; k; --k, ++i, ptr+=step, ptrwrap-=step) *ptrwrap += CONJ(*ptr);
#         xdbg<<"i = "<<i<<std::endl;
#         if (i == m) break;
#         *ptrwrap += CONJ(*ptr);
#         k = std::min(m-i, mwrap-1);
#         xdbg<<"k = "<<k<<std::endl;
#         if (step == 1)
#             for (; k; --k, ++i) *ptrwrap++ += *ptr++;
#         else
#             for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
#         xdbg<<"i = "<<i<<std::endl;
#         if (i == m) break;
#         *ptrwrap += *ptr;
#     }
# }

# template <typename T>
# void wrapImage(ImageView<T> im, const Bounds<int>& b, bool hermx, bool hermy)
# {
#     dbg<<"Start ImageView::wrap: b = "<<b<<std::endl;
#     dbg<<"self bounds = "<<im.getBounds()<<std::endl;
#     //set_verbose(2);

#     const int i1 = b.getXMin()-im.getBounds().getXMin();
#     const int i2 = b.getXMax()-im.getBounds().getXMin()+1;  // +1 for "1 past the end"
#     const int j1 = b.getYMin()-im.getBounds().getYMin();
#     const int j2 = b.getYMax()-im.getBounds().getYMin()+1;
#     xdbg<<"i1,i2,j1,j2 = "<<i1<<','<<i2<<','<<j1<<','<<j2<<std::endl;
#     const int mwrap = i2-i1;
#     const int nwrap = j2-j1;
#     const int skip = im.getNSkip();
#     const int step = im.getStep();
#     const int stride = im.getStride();
#     const int m = im.getNCol();
#     const int n = im.getNRow();
#     T* ptr = im.getData();

#     if (hermx) {
#         // In the hermitian x case, we need to wrap the columns first, otherwise the bookkeeping
#         // becomes difficult.
#         //
#         // Each row has a corresponding row that stores the conjugate information for the
#         // negative x values that are not stored.  We do these pairs of rows together.
#         //
#         // The exception is row 0 (which here is j==(n-1)/2), which is its own conjugate, so
#         // it works slightly differently.
#         assert(i1 == 0);

#         int mid = (n-1)/2;  // The value of j that corresponds to the j==0 in the normal notation.

#         T* ptr1 = im.getData() + (i2-1)*step;
#         T* ptr2 = im.getData() + (n-1)*stride + (i2-1)*step;

#         // These skips will take us from the end of one row to the i2-1 element in the next row.
#         int skip1 = skip + (i2-1)*step;
#         int skip2 = skip1 - 2*stride; // This is negative.  We add this value to ptr2.

#         for (int j=0; j<mid; ++j, ptr1+=skip1, ptr2+=skip2) {
#             xdbg<<"Wrap rows "<<j<<","<<n-j-1<<" into columns ["<<i1<<','<<i2<<")\n";
#             xdbg<<"ptrs = "<<ptr1-im.getData()<<"  "<<ptr2-im.getData()<<std::endl;
#             wrap_hermx_cols_pair(ptr1, ptr2, m, mwrap, step);
#         }
#         // Finally, the row that is really j=0 (but here is j=(n-1)/2) also needs to be wrapped
#         // singly.
#         xdbg<<"Wrap row "<<mid<<" into columns ["<<i1<<','<<i2<<")\n";
#         xdbg<<"ptrs = "<<ptr1-im.getData()<<"  "<<ptr2-im.getData()<<std::endl;
#         wrap_hermx_cols(ptr1, m, mwrap, step);
#     }

#     // If hermx is false, then we wrap the rows first instead.
#     if (hermy) {
#         assert(j1 == 0);
#         // In this case, the number of rows in the target image corresponds to N/2+1.
#         // Rows 0 and N/2 need special handling, since the wrapping is really for the
#         // range (-N/2,N/2], even though the negative rows are not stored.
#         // We start with row N/2 (aka j2-1), which needs to wrap its conjugate (-N/2) onto itself.
#         // Then as j increases, we decrease jj and continue wrapping conjugates.
#         // When we get to something that wraps onto row 0 (the first one will correspond to
#         // j=-N, which is the conjugate of what is stored at j=N), we need to repeat with
#         // a regular non-conjugate wrapping for the positive row (e.g. j=N itself) which also
#         // wraps onto row 0.
#         // Then we run jj back up wrapping normally, until we get to N/2 again (aka j2-1).
#         // The negative row will wrap normally onto -N/2, which means we need to also do a
#         // conjugate wrapping onto N/2.

#         // Start with j == jj = j2-1.
#         int jj = j2-1;
#         ptr += jj * stride;
#         T* ptrwrap = ptr + (m-1) * step;

#         // Do the first row separately, since we need to do it slightly differently, as
#         // we are overwriting the input data as we go, so we would double add it if we did
#         // it the normal way.
#         xdbg<<"Wrap first row "<<jj<<" onto row = "<<jj<<" using conjugation.\n";
#         xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
#         wrap_row_selfconj(ptr, ptrwrap, m, step);

#         ptr += skip;
#         ptrwrap -= skip;
#         --jj;
#         int  j= j2;
#         while (1) {
#             int k = std::min(n-j,jj);  // How many conjugate rows to do?
#             for (; k; --k, ++j, --jj, ptr+=skip, ptrwrap-=skip) {
#                 xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<" using conjugation.\n";
#                 xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
#                 wrap_row_conj(ptr, ptrwrap, m, step);
#             }
#             assert(j==n || jj == j1);
#             if (j == n) break;
#             assert(j < n);
#             // On the last one, don't increment ptrs, since we need to repeat with the non-conj add.
#             wrap_row_conj(ptr, ptrwrap, m, step);
#             ptr -= m*step;
#             ptrwrap += step;

#             k = std::min(n-j,nwrap-1);  // How many non-conjugate rows to do?
#             for (; k; --k, ++j, ++jj, ptr+=skip, ptrwrap+=skip) {
#                 xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<std::endl;
#                 xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
#                 wrap_row(ptr, ptrwrap, m, step);
#             }
#             assert(j==n || jj == j2-1);
#             if (j == n) break;
#             assert(j < n);
#             wrap_row(ptr, ptrwrap, m, step);
#             ptr -= m*step;
#             ptrwrap -= step;
#         }
#     } else {
#         // The regular case is mostly simpler (no conjugate stuff to worry about).
#         // However, we don't have the luxury of knowing that j1==0, so we need to start with
#         // the rows j<j1, then skip over [j1,j2) when we get there and continue with j>=j2.

#         // Row 0 maps onto j2 - (j2 % nwrap) (although we may need to subtract nwrap).
#         int jj = j2 - (j2 % nwrap);
#         if (jj == j2) jj = j1;
#         T* ptrwrap = ptr + jj * stride;
#         for (int j=0; j<n;) {
#             // When we get here, we can just skip to j2 and keep going.
#             if (j == j1) {
#                 assert(ptr == ptrwrap);
#                 j = j2;
#                 ptr += nwrap * stride;
#             }
#             int k = std::min(n-j,j2-jj);  // How many to do before looping back.
#             for (; k; --k, ++j, ++jj, ptr+=skip, ptrwrap+=skip) {
#                 xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<std::endl;
#                 xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
#                 wrap_row(ptr, ptrwrap, m, step);
#             }
#             jj = j1;
#             ptrwrap -= nwrap * stride;
#         }
#     }

#     // In the normal (not hermx) case, we now wrap rows [j1,j2) into the columns [i1,i2).
#     if (!hermx) {
#         ptr = im.getData() + j1*stride;
#         for (int j=j1; j<j2; ++j, ptr+=skip) {
#             xdbg<<"Wrap row "<<j<<" into columns ["<<i1<<','<<i2<<")\n";
#             xdbg<<"ptr = "<<ptr-im.getData()<<std::endl;
#             wrap_cols(ptr, m, mwrap, i1, i2, step);
#         }
#     }
# }

# """
