#include <math.h>
#ifdef ceild
# undef ceild
#endif
#ifdef floord
# undef floord
#endif
#ifdef max
# undef max
#endif
#ifdef min
# undef min
#endif
#define ceild(n,d) (((n) < 0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(x,y) (((x) < 0)? -((-(x)+(y)-1)/(y)) : (x)/(y))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  register int lbv, ubv, lb, ub, lb1, ub1, lb2, ub2;
  register int c1, c2, c3;
#pragma scop
if ((self.NI >= 1) && (self.NJ >= 1)) {
  for (c1 = 0; c1 <= (self.NI + -1); c1++) {
    {
      for (c2 = 0; c2 <= (self.NJ + -1); c2++) {
        {
          C[c1][c2]=(C[c1][c2]*beta);
        }
      }
    }
  }
  if (self.NK >= 1) {
    for (c1 = 0; c1 <= (self.NI + -1); c1++) {
      {
        for (c2 = 0; c2 <= (self.NJ + -1); c2++) {
          {
            for (c3 = 0; c3 <= (self.NK + -1); c3++) {
              {
                C[c1][c2]=(C[c1][c2]+((alpha*A[c1][c3])*B[c3][c2]));
              }
            }
          }
        }
      }
    }
  }
}
#pragma endscop
