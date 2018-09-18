/*********************************************************************************
Copyright (c) 2015, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************/


#include "common.h"
#include <immintrin.h>

#if 0
 
#define OLD_M	%rdi
#define OLD_N	%rsi
#define M	%r13
#define J	%r14
#define OLD_K	%rdx

#define A	%rcx
#define B	%r8
#define C	%r9
#define LDC	%r10
	
#define I	%r11
#define AO	%rdi
#define BO	%rsi
#define	CO1	%r15
#define K	%r12
#define	SP	%rbx

#define BO1	%rdi
#define BO2	%r15
#define BO3	%rbp

#ifndef WINDOWS_ABI

#define STACKSIZE 96

#else

#define STACKSIZE 256
#define L_BUFFER_SIZE 128*8*12+512

#define OLD_A		40 + STACKSIZE(%rsp)
#define OLD_B		48 + STACKSIZE(%rsp)
#define OLD_C		56 + STACKSIZE(%rsp)
#define OLD_LDC		64 + STACKSIZE(%rsp)
#define OLD_OFFSET	72 + STACKSIZE(%rsp)

#endif


#define Ndiv12	 24(%rsp)
#define Nmod12	 32(%rsp)
#define N	 40(%rsp)
#define ALPHA	 48(%rsp)
#define OFFSET	 56(%rsp)
#define KK	 64(%rsp)
#define KKK	 72(%rsp)
#define BUFFER1	           128(%rsp)

#if defined(OS_WINDOWS)
#if   L_BUFFER_SIZE > 16384
#define STACK_TOUCH \
        movl    $ 0,  4096 * 4(%rsp);\
        movl    $ 0,  4096 * 3(%rsp);\
        movl    $ 0,  4096 * 2(%rsp);\
        movl    $ 0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 12288
#define STACK_TOUCH \
        movl    $ 0,  4096 * 3(%rsp);\
        movl    $ 0,  4096 * 2(%rsp);\
        movl    $ 0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 8192
#define STACK_TOUCH \
        movl    $ 0,  4096 * 2(%rsp);\
        movl    $ 0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 4096
#define STACK_TOUCH \
        movl    $ 0,  4096 * 1(%rsp);
#else
#define STACK_TOUCH
#endif
#else
#define STACK_TOUCH
#endif
#endif

#define	A_PR1	512
#define	B_PR1	512
#define L_BUFFER_SIZE 256*8*12+4096

/*******************************************************************************************
* Macro definitions
*******************************************************************************************/


static inline __m256d _mm256_blend_pd_gnu(__m256d A, __m256d B, int i)
{ 
	return _mm256_blend_pd(B,A, i);
}

static void print_ymm(char *str, __m256d ymm)
{
	printf("%s   %5.1f\t%5.1f\t%5.1f\t%5.1f\n", str, ymm[0], ymm[1], ymm[2], ymm[3]);
}

#define  INIT4x12()				\
	ymm4 = _mm256_setzero_pd();		\
	ymm5 = _mm256_setzero_pd();		\
	ymm6 = _mm256_setzero_pd();		\
	ymm7 = _mm256_setzero_pd();		\
	ymm8 = _mm256_setzero_pd();		\
	ymm9 = _mm256_setzero_pd();		\
	ymm10 = _mm256_setzero_pd();		\
	ymm11 = _mm256_setzero_pd();		\
	ymm12 = _mm256_setzero_pd();		\
	ymm13 = _mm256_setzero_pd();		\
	ymm14 = _mm256_setzero_pd();		\
	ymm15 = _mm256_setzero_pd();		



#define KERNEL4x12_I()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
	ymm3  = _mm256_loadu_pd(BO - 4);	\
						\
	ymm4  = ymm0 * ymm1;			\
	ymm8  = ymm0 * ymm2;			\
	ymm12 = ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = ymm0 * ymm1;			\
	ymm9  = ymm0 * ymm2;			\
	ymm13 = ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = ymm0 * ymm1;			\
	ymm10 = ymm0 * ymm2;			\
	ymm14 = ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = ymm0 * ymm1;			\
	ymm11 = ymm0 * ymm2;			\
	ymm15 = ymm0 * ymm3;			\
						\
	BO   += 12;				\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
	ymm3  = _mm256_loadu_pd(BO - 4);	


#define KERNEL4x12_M1() 			\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
						\
	ymm4  = _mm256_fmadd_pd(ymm0, ymm1, ymm4);		\
	ymm8  = _mm256_fmadd_pd(ymm0, ymm2, ymm8);		\
	ymm12 = _mm256_fmadd_pd(ymm0, ymm3, ymm12);		\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = _mm256_fmadd_pd(ymm0, ymm1, ymm5);	\
	ymm9  = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
	ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
	ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
	ymm15 = _mm256_fmadd_pd(ymm0, ymm3, ymm15);			\
						\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
	ymm3  = _mm256_loadu_pd(BO - 4);


#define KERNEL4x12_M2()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4  = _mm256_fmadd_pd(ymm0, ymm1, ymm4);			\
	ymm8  = _mm256_fmadd_pd(ymm0, ymm2, ymm8);			\
	ymm12 = _mm256_fmadd_pd(ymm0, ymm3, ymm12);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = _mm256_fmadd_pd(ymm0, ymm1, ymm5);			\
	ymm9  = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
	ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
	ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
	ymm15 = _mm256_fmadd_pd(ymm0, ymm3, ymm15);			\
						\
	AO += 8;				\
	ymm1  = _mm256_loadu_pd(BO + 0);	\
	ymm2  = _mm256_loadu_pd(BO + 4);	\
	ymm3  = _mm256_loadu_pd(BO + 8);	\
	BO   += 24;				



#define KERNEL4x12_E()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4  = _mm256_fmadd_pd(ymm0, ymm1, ymm4);			\
	ymm8  = _mm256_fmadd_pd(ymm0, ymm2, ymm8);			\
	ymm12 = _mm256_fmadd_pd(ymm0, ymm3, ymm12);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = _mm256_fmadd_pd(ymm0, ymm1, ymm5);			\
	ymm9  = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
	ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
	ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
	ymm15 = _mm256_fmadd_pd(ymm0, ymm3, ymm15);			\
						\
	AO += 8;				\
	BO += 12;				

#define KERNEL4x12_SUB()			\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
	ymm3  = _mm256_loadu_pd(BO - 4);	\
						\
	ymm4  += ymm0 * ymm1;			\
	ymm8  += ymm0 * ymm2;			\
	ymm12 += ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  += ymm0 * ymm1;			\
	ymm9  += ymm0 * ymm2;			\
	ymm13 += ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  += ymm0 * ymm1;			\
	ymm10 += ymm0 * ymm2;			\
	ymm14 += ymm0 * ymm3;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  += ymm0 * ymm1;			\
	ymm11 += ymm0 * ymm2;			\
	ymm15 += ymm0 * ymm3;			\
						\
	AO += 4;				\
	BO += 12;				


#define  SAVE4x12(ALPHA)				\
	ymm0 = _mm256_set1_pd(ALPHA);			\
	ymm4 *= ymm0;					\
	ymm5 *= ymm0;					\
	ymm6 *= ymm0;					\
	ymm7 *= ymm0;					\
	ymm8 *= ymm0;					\
	ymm9 *= ymm0;					\
	ymm10 *= ymm0;					\
	ymm11 *= ymm0;					\
	ymm12 *= ymm0;					\
	ymm13 *= ymm0;					\
	ymm14 *= ymm0;					\
	ymm15 *= ymm0;					\
							\
	ymm5  = _mm256_permute4x64_pd(ymm5, 0xb1);		\
	ymm7  = _mm256_permute4x64_pd(ymm7, 0xb1);		\
							\
	ymm0  = _mm256_blend_pd_gnu(ymm5, ymm4, 0x0a);	\
	ymm1  = _mm256_blend_pd_gnu(ymm5, ymm4, 0x05);	\
	ymm2  = _mm256_blend_pd_gnu(ymm7, ymm6, 0x0a);	\
	ymm3  = _mm256_blend_pd_gnu(ymm7, ymm6, 0x05);	\
							\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4  = _mm256_blend_pd_gnu(ymm0, ymm2, 0x03);	\
	ymm5  = _mm256_blend_pd_gnu(ymm1, ymm3, 0x03);	\
	ymm6  = _mm256_blend_pd_gnu(ymm2, ymm0, 0x03);	\
	ymm7  = _mm256_blend_pd_gnu(ymm3, ymm1, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (0 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (1 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (2 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (3 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (0 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (1 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (2 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (3 * LDC)/8, ymm7);	\
							\
	ymm9  = _mm256_permute4x64_pd(ymm9, 0xb1);		\
	ymm11 = _mm256_permute4x64_pd(ymm11, 0xb1);		\
							\
	ymm0  = _mm256_blend_pd_gnu(ymm9, ymm8, 0x0a);	\
	ymm1  = _mm256_blend_pd_gnu(ymm9, ymm8, 0x05);	\
	ymm2  = _mm256_blend_pd_gnu(ymm11, ymm10, 0x0a);	\
	ymm3  = _mm256_blend_pd_gnu(ymm11, ymm10, 0x05);	\
							\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4  = _mm256_blend_pd_gnu(ymm0, ymm2, 0x03);	\
	ymm5  = _mm256_blend_pd_gnu(ymm1, ymm3, 0x03);	\
	ymm6  = _mm256_blend_pd_gnu(ymm2, ymm0, 0x03);	\
	ymm7  = _mm256_blend_pd_gnu(ymm3, ymm1, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (4 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (5 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (6 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (7 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (4 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (5 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (6 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (7 * LDC)/8, ymm7);	\
							\
	ymm13 = _mm256_permute4x64_pd(ymm13, 0xb1);		\
	ymm15 = _mm256_permute4x64_pd(ymm15, 0xb1);		\
	 						\
	ymm0  = _mm256_blend_pd_gnu(ymm13, ymm12, 0x0a);	\
	ymm1  = _mm256_blend_pd_gnu(ymm13, ymm12, 0x05);	\
	ymm2  = _mm256_blend_pd_gnu(ymm15, ymm14, 0x0a);	\
	ymm3  = _mm256_blend_pd_gnu(ymm15, ymm14, 0x05);	\
							\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2  = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3  = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4  = _mm256_blend_pd_gnu(ymm0, ymm2, 0x03);	\
	ymm5  = _mm256_blend_pd_gnu(ymm1, ymm3, 0x03);	\
	ymm6  = _mm256_blend_pd_gnu(ymm2, ymm0, 0x03);	\
	ymm7  = _mm256_blend_pd_gnu(ymm3, ymm1, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (8 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (9 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (10 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (11 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (8 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (9 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (10 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (11 * LDC)/8, ymm7);	\
							\
	CO1 += 4;



/******************************************************************************************/


#define INIT2x12() 				\
	xmm4 = _mm_setzero_pd(); 		\
	xmm5 = _mm_setzero_pd(); 		\
	xmm6 = _mm_setzero_pd(); 		\
	xmm7 = _mm_setzero_pd(); 		\
	xmm8 = _mm_setzero_pd(); 		\
	xmm9 = _mm_setzero_pd(); 		\
	xmm10 = _mm_setzero_pd(); 		\
	xmm11 = _mm_setzero_pd(); 		\
	xmm12 = _mm_setzero_pd(); 		\
	xmm13 = _mm_setzero_pd(); 		\
	xmm14 = _mm_setzero_pd(); 		\
	xmm15 = _mm_setzero_pd(); 


#define KERNEL2x12_SUB() 			\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm1 = _mm_set1_pd(*(BO - 12));		\
	xmm2 = _mm_set1_pd(*(BO - 11));		\
	xmm3 = _mm_set1_pd(*(BO - 10));		\
	xmm4 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 9));		\
	xmm5 += xmm0 * xmm2;			\
	xmm2 = _mm_set1_pd(*(BO - 8));		\
	xmm6 += xmm0 * xmm3;			\
	xmm3 = _mm_set1_pd(*(BO - 7));		\
	xmm7 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 6));		\
	xmm8 += xmm0 * xmm2;			\
	xmm2 = _mm_set1_pd(*(BO - 5));		\
	xmm9 += xmm0 * xmm3;			\
	xmm3 = _mm_set1_pd(*(BO - 4));		\
	xmm10 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 3));		\
	xmm11 += xmm0 * xmm2;			\
	xmm2 = _mm_set1_pd(*(BO - 2));		\
	xmm12 += xmm0 * xmm3;			\
	xmm3 = _mm_set1_pd(*(BO - 1));		\
	xmm13 += xmm0 * xmm1;			\
	xmm14 += xmm0 * xmm2;			\
	xmm15 += xmm0 * xmm3;			\
	BO += 12;				\
	AO += 2;


#define SAVE2x12(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
	xmm5 *= xmm0;					\
	xmm6 *= xmm0;					\
	xmm7 *= xmm0;					\
	xmm8 *= xmm0;					\
	xmm9 *= xmm0;					\
	xmm10 *= xmm0;					\
	xmm11 *= xmm0;					\
	xmm12 *= xmm0;					\
	xmm13 *= xmm0;					\
	xmm14 *= xmm0;					\
	xmm15 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1 + (0 * LDC)/8);	\
	xmm5 += _mm_loadu_pd(CO1 + (1 * LDC)/8);	\
	xmm6 += _mm_loadu_pd(CO1 + (2 * LDC)/8);	\
	xmm7 += _mm_loadu_pd(CO1 + (3 * LDC)/8);	\
							\
	_mm_storeu_pd(CO1 + (0 * LDC)/8, xmm4);		\
	_mm_storeu_pd(CO1 + (1 * LDC)/8, xmm5);		\
	_mm_storeu_pd(CO1 + (2 * LDC)/8, xmm6);		\
	_mm_storeu_pd(CO1 + (3 * LDC)/8, xmm7);		\
							\
	xmm8 += _mm_loadu_pd(CO1 + (4 * LDC)/8);	\
	xmm9 += _mm_loadu_pd(CO1 + (5 * LDC)/8);	\
	xmm10+= _mm_loadu_pd(CO1 + (6 * LDC)/8);	\
	xmm11+= _mm_loadu_pd(CO1 + (7 * LDC)/8);	\
							\
	_mm_storeu_pd(CO1 + (4 * LDC)/8, xmm8);		\
	_mm_storeu_pd(CO1 + (5 * LDC)/8, xmm9);		\
	_mm_storeu_pd(CO1 + (6 * LDC)/8, xmm10);	\
	_mm_storeu_pd(CO1 + (7 * LDC)/8, xmm11);	\
							\
	xmm12 += _mm_loadu_pd(CO1 + ( 8 * LDC)/8);	\
	xmm13 += _mm_loadu_pd(CO1 + ( 9 * LDC)/8);	\
	xmm14 += _mm_loadu_pd(CO1 + (10 * LDC)/8);	\
	xmm15 += _mm_loadu_pd(CO1 + (11 * LDC)/8);	\
							\
	_mm_storeu_pd(CO1 + ( 8 * LDC)/8, xmm12);	\
	_mm_storeu_pd(CO1 + ( 9 * LDC)/8, xmm13);	\
	_mm_storeu_pd(CO1 + (10 * LDC)/8, xmm14);	\
	_mm_storeu_pd(CO1 + (11 * LDC)/8, xmm15);	\
	CO1 += 2;


/******************************************************************************************/

#define INIT1x12()		\
	dbl4 = 0; 		\
	dbl5 = 0;  		\
	dbl6 = 0;  		\
	dbl7 = 0; 		\
	dbl8 = 0;  		\
	dbl9 = 0;  		\
	dbl10 = 0; 		\
	dbl11 = 0;  		\
	dbl12 = 0;  		\
	dbl13 = 0; 		\
	dbl14 = 0;  		\
	dbl15 = 0;  		


#define KERNEL1x12_SUB() 			\
	dbl0 = *(AO - 16);			\
	dbl1 = *(BO - 12);			\
	dbl2 = *(BO - 11);			\
	dbl3 = *(BO - 10);			\
	dbl4 += dbl0 * dbl1;			\
	dbl1 = *(BO - 9);			\
	dbl5 += dbl0 * dbl2;			\
	dbl2 = *(BO - 8);			\
	dbl6 += dbl0 * dbl3;			\
	dbl3 = *(BO - 7);			\
	dbl7 += dbl0 * dbl1;			\
	dbl1 = *(BO - 6);			\
	dbl8 += dbl0 * dbl2;			\
	dbl2 = *(BO - 5);			\
	dbl9 += dbl0 * dbl3;			\
	dbl3 = *(BO - 4);			\
	dbl10 += dbl0 * dbl1;			\
	dbl1 = *(BO - 3);			\
	dbl11 += dbl0 * dbl2;			\
	dbl2 = *(BO - 2);			\
	dbl12 += dbl0 * dbl3;			\
	dbl3 = *(BO - 1);			\
	dbl13 += dbl0 * dbl1;			\
	dbl14 += dbl0 * dbl2;			\
	dbl15 += dbl0 * dbl3;			\
	BO += 12;				\
	AO += 1;


#define SAVE1x12(ALPHA)				\
	dbl0 = ALPHA;				\
	dbl4 *= dbl0;				\
	dbl5 *= dbl0;				\
	dbl6 *= dbl0;				\
	dbl7 *= dbl0;				\
	dbl8 *= dbl0;				\
	dbl9 *= dbl0;				\
	dbl10 *= dbl0;				\
	dbl11 *= dbl0;				\
	dbl12 *= dbl0;				\
	dbl13 *= dbl0;				\
	dbl14 *= dbl0;				\
	dbl15 *= dbl0;				\
						\
	dbl4 += *(CO1 + (0 * LDC)/8);		\
	dbl5 += *(CO1 + (1 * LDC)/8);		\
	dbl6 += *(CO1 + (2 * LDC)/8);		\
	dbl7 += *(CO1 + (3 * LDC)/8);		\
	*(CO1 + (0 * LDC)/8) = dbl4;		\
	*(CO1 + (1 * LDC)/8) = dbl5;		\
	*(CO1 + (2 * LDC)/8) = dbl6;		\
	*(CO1 + (3 * LDC)/8) = dbl7;		\
						\
	dbl8  += *(CO1 + (4 * LDC)/8);		\
	dbl9  += *(CO1 + (5 * LDC)/8);		\
	dbl10 += *(CO1 + (6 * LDC)/8);		\
	dbl11 += *(CO1 + (7 * LDC)/8);		\
	*(CO1 + (4 * LDC)/8) = dbl8;		\
	*(CO1 + (5 * LDC)/8) = dbl9;		\
	*(CO1 + (6 * LDC)/8) = dbl10;		\
	*(CO1 + (7 * LDC)/8) = dbl11;		\
						\
	dbl12 += *(CO1 + ( 8 * LDC)/8);		\
	dbl13 += *(CO1 + ( 9 * LDC)/8);		\
	dbl14 += *(CO1 + (10 * LDC)/8);		\
	dbl15 += *(CO1 + (11 * LDC)/8);		\
	*(CO1 + ( 8 * LDC)/8) = dbl12;		\
	*(CO1 + ( 9 * LDC)/8) = dbl13;		\
	*(CO1 + (10 * LDC)/8) = dbl14;		\
	*(CO1 + (11 * LDC)/8) = dbl15;		\
						\
	CO1 += 1;


/******************************************************************************************/


#define INIT4x8()				\
	ymm4 = _mm256_setzero_pd();		\
	ymm5 = _mm256_setzero_pd();		\
	ymm6 = _mm256_setzero_pd();		\
	ymm7 = _mm256_setzero_pd();		\
	ymm8 = _mm256_setzero_pd();		\
	ymm9 = _mm256_setzero_pd();		\
	ymm10 = _mm256_setzero_pd();		\
	ymm11 = _mm256_setzero_pd();		\


#define KERNEL4x8_I()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
						\
	ymm4  = ymm0 * ymm1;			\
	ymm8  = ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = ymm0 * ymm1;			\
	ymm9  = ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = ymm0 * ymm1;			\
	ymm10 = ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = ymm0 * ymm1;			\
	ymm11 = ymm0 * ymm2;			\
						\
	BO   += 8;				\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	


#define KERNEL4x8_M1()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
						\
	ymm4 = _mm256_fmadd_pd(ymm0, ymm1, ymm4);			\
	ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);			\
						\
	ymm0 = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 = _mm256_fmadd_pd(ymm0, ymm1, ymm5);			\
	ymm9 = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
						\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	

	

#define KERNEL4x8_M2()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4 = _mm256_fmadd_pd(ymm0, ymm1, ymm4);			\
	ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 = _mm256_fmadd_pd(ymm0, ymm1, ymm5);			\
	ymm9 = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
						\
	ymm1  = _mm256_loadu_pd(BO - 4);	\
	ymm2  = _mm256_loadu_pd(BO - 0);	\
						\
	AO   += 8;				\
	BO   += 16;


#define KERNEL4x8_E()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4 = _mm256_fmadd_pd(ymm0, ymm1, ymm4);			\
	ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);			\
						\
	ymm0 = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 = _mm256_fmadd_pd(ymm0, ymm1, ymm5);			\
	ymm9 = _mm256_fmadd_pd(ymm0, ymm2, ymm9);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = _mm256_fmadd_pd(ymm0, ymm1, ymm6);			\
	ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = _mm256_fmadd_pd(ymm0, ymm1, ymm7);			\
	ymm11 = _mm256_fmadd_pd(ymm0, ymm2, ymm11);			\
						\
	AO   += 8;				\
	BO   += 8;
	

#define KERNEL4x8_SUB()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
	ymm2  = _mm256_loadu_pd(BO - 8);	\
						\
	ymm4 += ymm0 * ymm1;			\
	ymm8 += ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 += ymm0 * ymm1;			\
	ymm9 += ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6 += ymm0 * ymm1;			\
	ymm10+= ymm0 * ymm2;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7 += ymm0 * ymm1;			\
	ymm11+= ymm0 * ymm2;			\
	AO += 4;				\
	BO += 8;


#define SAVE4x8(ALPHA)					\
	ymm0 = _mm256_set1_pd(ALPHA);			\
	ymm4 *= ymm0;					\
	ymm5 *= ymm0;					\
	ymm6 *= ymm0;					\
	ymm7 *= ymm0;					\
	ymm8 *= ymm0;					\
	ymm9 *= ymm0;					\
	ymm10 *= ymm0;					\
	ymm11 *= ymm0;					\
							\
	ymm5 = _mm256_permute4x64_pd(ymm5, 0xb1);		\
	ymm7 = _mm256_permute4x64_pd(ymm7, 0xb1);		\
							\
	ymm0 = _mm256_blend_pd_gnu(ymm5, ymm4, 0x0a);	\
	ymm1 = _mm256_blend_pd_gnu(ymm5, ymm4, 0x05);	\
	ymm2 = _mm256_blend_pd_gnu(ymm7, ymm6, 0x0a);	\
	ymm3 = _mm256_blend_pd_gnu(ymm7, ymm6, 0x05);	\
							\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4 = _mm256_blend_pd_gnu(ymm0, ymm2, 0x03);	\
	ymm5 = _mm256_blend_pd_gnu(ymm1, ymm3, 0x03);	\
	ymm6 = _mm256_blend_pd_gnu(ymm2, ymm0, 0x03);	\
	ymm7 = _mm256_blend_pd_gnu(ymm3, ymm1, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (0 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (1 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (2 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (3 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (0 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (1 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (2 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (3 * LDC)/8, ymm7);	\
							\
	ymm9 = _mm256_permute4x64_pd(ymm9, 0xb1);		\
	ymm11 = _mm256_permute4x64_pd(ymm11, 0xb1);		\
							\
	ymm0 = _mm256_blend_pd_gnu(ymm9, ymm8, 0x0a);	\
	ymm1 = _mm256_blend_pd_gnu(ymm9, ymm8, 0x05);	\
	ymm2 = _mm256_blend_pd_gnu(ymm11, ymm10, 0x0a);	\
	ymm3 = _mm256_blend_pd_gnu(ymm11, ymm10, 0x05);	\
							\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4 = _mm256_blend_pd_gnu(ymm0, ymm2, 0x03);	\
	ymm5 = _mm256_blend_pd_gnu(ymm1, ymm3, 0x03);	\
	ymm6 = _mm256_blend_pd_gnu(ymm2, ymm0, 0x03);	\
	ymm7 = _mm256_blend_pd_gnu(ymm3, ymm1, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (4 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (5 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (6 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (7 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (4 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (5 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (6 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (7 * LDC)/8, ymm7);	\
							\
	CO1 += 4;

/******************************************************************************************/

#define INIT2x8()				\
	xmm4 = _mm_setzero_pd(); 		\
	xmm5 = _mm_setzero_pd(); 		\
	xmm6 = _mm_setzero_pd(); 		\
	xmm7 = _mm_setzero_pd(); 		\
	xmm8 = _mm_setzero_pd(); 		\
	xmm9 = _mm_setzero_pd(); 		\
	xmm10 = _mm_setzero_pd(); 		\
	xmm11 = _mm_setzero_pd(); 		\


#define KERNEL2x8_SUB()				\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm1 = _mm_set1_pd(*(BO - 12));		\
	xmm2 = _mm_set1_pd(*(BO - 11));		\
	xmm3 = _mm_set1_pd(*(BO - 10));		\
	xmm4 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 9));		\
	xmm5 += xmm0 * xmm2;			\
	xmm2 = _mm_set1_pd(*(BO - 8));		\
	xmm6 += xmm0 * xmm3;			\
	xmm3 = _mm_set1_pd(*(BO - 7));		\
	xmm7 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 6));		\
	xmm8 += xmm0 * xmm2;			\
	xmm2 = _mm_set1_pd(*(BO - 5));		\
	xmm9 += xmm0 * xmm3;			\
	xmm10 += xmm0 * xmm1;			\
	xmm11 += xmm0 * xmm2;			\
	BO += 8;				\
	AO += 2;

#define  SAVE2x8(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
	xmm5 *= xmm0;					\
	xmm6 *= xmm0;					\
	xmm7 *= xmm0;					\
	xmm8 *= xmm0;					\
	xmm9 *= xmm0;					\
	xmm10 *= xmm0;					\
	xmm11 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1 + (0 * LDC)/8);	\
	xmm5 += _mm_loadu_pd(CO1 + (1 * LDC)/8);	\
	xmm6 += _mm_loadu_pd(CO1 + (2 * LDC)/8);	\
	xmm7 += _mm_loadu_pd(CO1 + (3 * LDC)/8);	\
							\
	_mm_storeu_pd(CO1 + (0 * LDC)/8, xmm4);		\
	_mm_storeu_pd(CO1 + (1 * LDC)/8, xmm5);		\
	_mm_storeu_pd(CO1 + (2 * LDC)/8, xmm6);		\
	_mm_storeu_pd(CO1 + (3 * LDC)/8, xmm7);		\
							\
	xmm8 += _mm_loadu_pd(CO1 + (4 * LDC)/8);	\
	xmm9 += _mm_loadu_pd(CO1 + (5 * LDC)/8);	\
	xmm10+= _mm_loadu_pd(CO1 + (6 * LDC)/8);	\
	xmm11+= _mm_loadu_pd(CO1 + (7 * LDC)/8);	\
	_mm_storeu_pd(CO1 + (4 * LDC)/8, xmm8);		\
	_mm_storeu_pd(CO1 + (5 * LDC)/8, xmm9);		\
	_mm_storeu_pd(CO1 + (6 * LDC)/8, xmm10);	\
	_mm_storeu_pd(CO1 + (7 * LDC)/8, xmm11);	\
	CO1 += 2;




/******************************************************************************************/

#define INIT1x8()				\
	dbl4 = 0;	\
	dbl5 = 0;	\
	dbl6 = 0;	\
	dbl7 = 0;	\
	dbl8 = 0;	\
	dbl9 = 0;	\
	dbl10 = 0;	\
	dbl11 = 0;	


#define KERNEL1x8_SUB()				\
	dbl0 = *(AO - 16);			\
	dbl1 = *(BO - 12);			\
	dbl2 = *(BO - 11);			\
	dbl3 = *(BO - 10);			\
	dbl4 += dbl0 * dbl1;			\
	dbl1 = *(BO - 9);			\
	dbl5 += dbl0 * dbl2;			\
	dbl2 = *(BO - 8);			\
	dbl6 += dbl0 * dbl3;			\
	dbl3 = *(BO - 7);			\
	dbl7 += dbl0 * dbl1;			\
	dbl1 = *(BO - 6);			\
	dbl8 += dbl0 * dbl2;			\
	dbl2 = *(BO - 5);			\
	dbl9  += dbl0 * dbl3;			\
	dbl10 += dbl0 * dbl1;			\
	dbl11 += dbl0 * dbl2;			\
	BO += 8;				\
	AO += 1;


#define SAVE1x8(ALPHA)				\
	dbl0 = ALPHA;				\
	dbl4 *= dbl0;				\
	dbl5 *= dbl0;				\
	dbl6 *= dbl0;				\
	dbl7 *= dbl0;				\
	dbl8 *= dbl0;				\
	dbl9 *= dbl0;				\
	dbl10 *= dbl0;				\
	dbl11 *= dbl0;				\
						\
	dbl4 += *(CO1 + (0 * LDC)/8);		\
	dbl5 += *(CO1 + (1 * LDC)/8);		\
	dbl6 += *(CO1 + (2 * LDC)/8);		\
	dbl7 += *(CO1 + (3 * LDC)/8);		\
	*(CO1 + (0 * LDC)/8) = dbl4;		\
	*(CO1 + (1 * LDC)/8) = dbl5;		\
	*(CO1 + (2 * LDC)/8) = dbl6;		\
	*(CO1 + (3 * LDC)/8) = dbl7;		\
						\
	dbl8  += *(CO1 + (4 * LDC)/8);		\
	dbl9  += *(CO1 + (5 * LDC)/8);		\
	dbl10 += *(CO1 + (6 * LDC)/8);		\
	dbl11 += *(CO1 + (7 * LDC)/8);		\
	*(CO1 + (4 * LDC)/8) = dbl8;		\
	*(CO1 + (5 * LDC)/8) = dbl9;		\
	*(CO1 + (6 * LDC)/8) = dbl10;		\
	*(CO1 + (7 * LDC)/8) = dbl11;		\
						\
	CO1 += 1;






/******************************************************************************************/

#define INIT4x4()				\
	ymm4 = _mm256_setzero_pd();		\
	ymm5 = _mm256_setzero_pd();		\
	ymm6 = _mm256_setzero_pd();		\
	ymm7 = _mm256_setzero_pd();		\


#define KERNEL4x4_I()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
						\
	ymm4  = ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5  = ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6  = ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7  = ymm0 * ymm1;			\
						\
	BO   += 4;				\
	ymm1  = _mm256_loadu_pd(BO - 12);	

#define KERNEL4x4_M1()				\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
						\
	ymm4 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7 += ymm0 * ymm1;			\
						\
	ymm1  = _mm256_loadu_pd(BO - 12);	\


#define KERNEL4x4_M2()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7 += ymm0 * ymm1;			\
						\
	ymm1  = _mm256_loadu_pd(BO - 8);	\
						\
	AO   += 8;				\
	BO   += 8;


#define KERNEL4x4_E()				\
	ymm0  = _mm256_loadu_pd(AO - 12);	\
						\
	ymm4 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7 += ymm0 * ymm1;			\
						\
	AO   += 8;				\
	BO   += 4;
	

#define KERNEL4x4_SUB() 			\
	ymm0  = _mm256_loadu_pd(AO - 16);	\
	ymm1  = _mm256_loadu_pd(BO - 12);	\
						\
	ymm4 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm5 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0x1b);	\
	ymm6 += ymm0 * ymm1;			\
						\
	ymm0  = _mm256_permute4x64_pd(ymm0, 0xb1);	\
	ymm7 += ymm0 * ymm1;			\
	AO += 4;				\
	BO += 4;


#define SAVE4x4(ALPHA)					\
	ymm0 = _mm256_set1_pd(ALPHA);			\
	ymm4 *= ymm0;					\
	ymm5 *= ymm0;					\
	ymm6 *= ymm0;					\
	ymm7 *= ymm0;					\
							\
	ymm5 = _mm256_permute4x64_pd(ymm5, 0xb1);		\
	ymm7 = _mm256_permute4x64_pd(ymm7, 0xb1);		\
							\
	ymm0 = _mm256_blend_pd(ymm4, ymm5, 0x0a);	\
	ymm1 = _mm256_blend_pd(ymm4, ymm5, 0x05);	\
	ymm2 = _mm256_blend_pd(ymm6, ymm7, 0x0a);	\
	ymm3 = _mm256_blend_pd(ymm6, ymm7, 0x05);	\
							\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0x1b);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0x1b);		\
	ymm2 = _mm256_permute4x64_pd(ymm2, 0xb1);		\
	ymm3 = _mm256_permute4x64_pd(ymm3, 0xb1);		\
							\
	ymm4 = _mm256_blend_pd(ymm2, ymm0, 0x03);	\
	ymm5 = _mm256_blend_pd(ymm3, ymm1, 0x03);	\
	ymm6 = _mm256_blend_pd(ymm0, ymm2, 0x03);	\
	ymm7 = _mm256_blend_pd(ymm1, ymm3, 0x03);	\
							\
	ymm4 += _mm256_loadu_pd(CO1 + (0 * LDC)/8);	\
	ymm5 += _mm256_loadu_pd(CO1 + (1 * LDC)/8);	\
	ymm6 += _mm256_loadu_pd(CO1 + (2 * LDC)/8);	\
	ymm7 += _mm256_loadu_pd(CO1 + (3 * LDC)/8);	\
	_mm256_storeu_pd(CO1 + (0 * LDC)/8, ymm4);	\
	_mm256_storeu_pd(CO1 + (1 * LDC)/8, ymm5);	\
	_mm256_storeu_pd(CO1 + (2 * LDC)/8, ymm6);	\
	_mm256_storeu_pd(CO1 + (3 * LDC)/8, ymm7);	\
							\
	CO1 += 4;


/******************************************************************************************/
/******************************************************************************************/

#define  INIT2x4()				\
	xmm4 = _mm_setzero_pd(); 		\
	xmm5 = _mm_setzero_pd(); 		\
	xmm6 = _mm_setzero_pd(); 		\
	xmm7 = _mm_setzero_pd(); 		\



#define KERNEL2x4_SUB()				\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm1 = _mm_set1_pd(*(BO - 12));		\
	xmm2 = _mm_set1_pd(*(BO - 11));		\
	xmm3 = _mm_set1_pd(*(BO - 10));		\
	xmm4 += xmm0 * xmm1;			\
	xmm1 = _mm_set1_pd(*(BO - 9));		\
	xmm5 += xmm0 * xmm2;			\
	xmm6 += xmm0 * xmm3;			\
	xmm7 += xmm0 * xmm1;			\
	BO += 4;				\
	AO += 2;



#define  SAVE2x4(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
	xmm5 *= xmm0;					\
	xmm6 *= xmm0;					\
	xmm7 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1 + (0 * LDC)/8);	\
	xmm5 += _mm_loadu_pd(CO1 + (1 * LDC)/8);	\
	xmm6 += _mm_loadu_pd(CO1 + (2 * LDC)/8);	\
	xmm7 += _mm_loadu_pd(CO1 + (3 * LDC)/8);	\
							\
	_mm_storeu_pd(CO1 + (0 * LDC)/8, xmm4);		\
	_mm_storeu_pd(CO1 + (1 * LDC)/8, xmm5);		\
	_mm_storeu_pd(CO1 + (2 * LDC)/8, xmm6);		\
	_mm_storeu_pd(CO1 + (3 * LDC)/8, xmm7);		\
							\
	CO1 += 2;

/******************************************************************************************/
/******************************************************************************************/

#define  INIT1x4()				\
	dbl4 = 0; 		\
	dbl5 = 0; 		\
	dbl6 = 0; 		\
	dbl7 = 0; 		\

#define KERNEL1x4_SUB()				\
	dbl0 = *(AO - 16);			\
	dbl1 = *(BO - 12);			\
	dbl2 = *(BO - 11);			\
	dbl3 = *(BO - 10);			\
	dbl8  = *(BO - 9);			\
						\
	dbl4 += dbl0 * dbl1;			\
	dbl5 += dbl0 * dbl2;			\
	dbl6 += dbl0 * dbl3;			\
	dbl7 += dbl0 * dbl8;			\
	BO += 4;				\
	AO += 1;


#define SAVE1x4(ALPHA)				\
	dbl0 = ALPHA;				\
	dbl4 *= dbl0;				\
	dbl5 *= dbl0;				\
	dbl6 *= dbl0;				\
	dbl7 *= dbl0;				\
						\
	dbl4 += *(CO1 + (0 * LDC)/8);		\
	dbl5 += *(CO1 + (1 * LDC)/8);		\
	dbl6 += *(CO1 + (2 * LDC)/8);		\
	dbl7 += *(CO1 + (3 * LDC)/8);		\
	*(CO1 + (0 * LDC)/8) = dbl4;		\
	*(CO1 + (1 * LDC)/8) = dbl5;		\
	*(CO1 + (2 * LDC)/8) = dbl6;		\
	*(CO1 + (3 * LDC)/8) = dbl7;		\
						\
						\
	CO1 += 1;


/******************************************************************************************/
/******************************************************************************************/

#define  INIT4x2()				\
	xmm4 = _mm_setzero_pd(); 		\
	xmm5 = _mm_setzero_pd(); 		\
	xmm6 = _mm_setzero_pd(); 		\
	xmm7 = _mm_setzero_pd(); 		\


#define KERNEL4x2_SUB()				\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm1 = _mm_loadu_pd(AO - 14);		\
	xmm2 = _mm_set1_pd(*(BO - 12));		\
	xmm3 = _mm_set1_pd(*(BO - 11));		\
	xmm4 += xmm0 * xmm2;			\
	xmm5 += xmm1 * xmm2;			\
	xmm6 += xmm0 * xmm3;			\
	xmm7 += xmm1 * xmm3;			\
	BO += 2;				\
	AO += 4;



#define SAVE4x2(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
	xmm5 *= xmm0;					\
	xmm6 *= xmm0;					\
	xmm7 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1);			\
	xmm5 += _mm_loadu_pd(CO1 + 2);			\
	xmm6 += _mm_loadu_pd(CO1 + (LDC)/8);		\
	xmm7 += _mm_loadu_pd(CO1 + (LDC)/8 + 2);	\
							\
	_mm_storeu_pd(CO1, xmm4);			\
	_mm_storeu_pd(CO1 + 2, xmm5);			\
	_mm_storeu_pd(CO1 + LDC/8, xmm6);		\
	_mm_storeu_pd(CO1 + LDC/8 + 2, xmm7);		\
							\
	CO1 += 4;


/******************************************************************************************/
/******************************************************************************************/

#define  INIT2x2()				\
	xmm4 = _mm_setzero_pd(); 		\
	xmm6 = _mm_setzero_pd(); 		\



#define KERNEL2x2_SUB()				\
	xmm2 = _mm_set1_pd(*(BO - 12));		\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm3 = _mm_set1_pd(*(BO - 11));		\
	xmm4 += xmm0 * xmm2;			\
	xmm6 += xmm0 * xmm3;			\
	BO += 2;				\
	AO += 2;


#define  SAVE2x2(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
	xmm6 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1);			\
	xmm6 += _mm_loadu_pd(CO1 + (LDC)/8);		\
							\
	_mm_storeu_pd(CO1, xmm4);			\
	_mm_storeu_pd(CO1 + LDC/8, xmm6);		\
							\
	CO1 += 2;


/******************************************************************************************/
/******************************************************************************************/

#define INIT1x2()				\
	dbl4 = 0;				\
	dbl5 = 0;			


#define KERNEL1x2_SUB()				\
	dbl0 = *(AO - 16);			\
	dbl1 = *(BO - 12);			\
	dbl2 = *(BO - 11);			\
	dbl4 += dbl0 * dbl1;			\
	dbl5 += dbl0 * dbl2;			\
	BO += 2;				\
	AO += 1;


#define SAVE1x2(ALPHA)				\
	dbl0 = ALPHA;				\
	dbl4 *= dbl0;				\
	dbl5 *= dbl0;				\
						\
	dbl4 += *(CO1 + (0 * LDC)/8);		\
	dbl5 += *(CO1 + (1 * LDC)/8);		\
	*(CO1 + (0 * LDC)/8) = dbl4;		\
	*(CO1 + (1 * LDC)/8) = dbl5;		\
						\
						\
	CO1 += 1;



/******************************************************************************************/
/******************************************************************************************/

#define INIT4x1()				\
	ymm4 = _mm256_setzero_pd();		\
	ymm5 = _mm256_setzero_pd();		\
	ymm6 = _mm256_setzero_pd();		\
	ymm7 = _mm256_setzero_pd();		


#define KERNEL4x1()				\
	ymm0 =  _mm256_set1_pd(*(BO - 12));	\
	ymm1 =  _mm256_set1_pd(*(BO - 11));	\
	ymm2 =  _mm256_set1_pd(*(BO - 10));	\
	ymm3 =  _mm256_set1_pd(*(BO -  9));	\
	\
	ymm4 += _mm256_loadu_pd(AO - 16) * ymm0;		\
	ymm5 += _mm256_loadu_pd(AO - 12) * ymm1;		\
	\
	ymm0 =  _mm256_set1_pd(*(BO - 8));	\
	ymm1 =  _mm256_set1_pd(*(BO - 7));	\
	\
	ymm6 += _mm256_loadu_pd(AO - 8) * ymm2;		\
	ymm7 += _mm256_loadu_pd(AO - 4) * ymm3;		\
	\
	ymm2 =  _mm256_set1_pd(*(BO - 6));	\
	ymm3 =  _mm256_set1_pd(*(BO - 5));	\
	\
	ymm4 += _mm256_loadu_pd(AO + 0) * ymm0;		\
	ymm5 += _mm256_loadu_pd(AO + 4) * ymm1;		\
	ymm6 += _mm256_loadu_pd(AO + 8) * ymm2;		\
	ymm7 += _mm256_loadu_pd(AO + 12) * ymm3;		\
	\
	BO += 8;	\
	AO += 32;




#define KERNEL4x1_SUB() 					\
	ymm2 = _mm256_set1_pd(*(BO - 12));			\
	ymm0 = _mm256_loadu_pd(AO - 16);			\
	ymm4 += ymm0 * ymm2;					\
	BO += 1;						\
	AO += 4;


#define SAVE4x1(ALPHA)						\
	ymm0 = _mm256_set1_pd(ALPHA);				\
	ymm4 += ymm5;						\
	ymm6 += ymm7;						\
	ymm4 += ymm6;						\
	ymm4 *= ymm0;						\
								\
	ymm4 += _mm256_loadu_pd(CO1);				\
	_mm256_storeu_pd(CO1, ymm4);				\
	CO1 += 4;


/******************************************************************************************/
/******************************************************************************************/

#define INIT2x1()					\
	xmm4 = _mm_setzero_pd(); 		


#define KERNEL2x1_SUB()				\
	xmm2 = _mm_set1_pd(*(BO - 12));		\
	xmm0 = _mm_loadu_pd(AO - 16);		\
	xmm4 += xmm0 * xmm2;			\
	BO += 1;				\
	AO += 2;


#define  SAVE2x1(ALPHA)					\
	xmm0 = _mm_set1_pd(ALPHA);			\
	xmm4 *= xmm0;					\
							\
	xmm4 += _mm_loadu_pd(CO1);			\
							\
	_mm_storeu_pd(CO1, xmm4);			\
							\
	CO1 += 2;


/******************************************************************************************/
/******************************************************************************************/

#define INIT1x1()	\
	dbl4 = 0;

#define KERNEL1x1_SUB() \
	dbl1 = *(BO - 12);	\
	dbl0 = *(AO - 16);	\
	dbl4 += dbl0 * dbl1;	\
	BO += 1;		\
	AO += 1;

#define SAVE1x1(ALPHA)	\
	dbl0 = ALPHA;	\
	dbl4 *= dbl0; 	\
	dbl4 += *CO1;	\
	*CO1 = dbl4;	\
	CO1 += 1;


/*******************************************************************************************/

/* START */


int __attribute__ ((noinline))
dgemm_kernel(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double *A, double *B, double *C, BLASLONG ldc)
{
	unsigned long OLD_M = m, OLD_N =m , OLD_K = k;
	unsigned long LDC = ldc * sizeof(double);

	unsigned long M=m, N=n, K=k;

	double BUFFER1[L_BUFFER_SIZE];
	unsigned long J;

	int Ndiv24, Nmod24;

	
	if (OLD_M == 0)
		return 0;
	if (OLD_N == 0)
		return 0;
	if (OLD_K == 0)
		return 0;

	Ndiv24 = N / 24;
	Nmod24 = N % 24;

	J = Ndiv24;

	while (J != 0) {
		int i;
		double *BO1, *BO2, *BO, *BO3;
		double *CO1;
		double *AO;

		BO1 = B;
		BO2 = B + K * 8;
		B = BO2;
	
		BO = BUFFER1; 
		
		i = K;
		do {
			__m256d t1, t2, t3;

			t1 = _mm256_loadu_pd(BO1);
			t2 = _mm256_loadu_pd(BO1 + 4);
			t3 = _mm256_loadu_pd(BO2);

			_mm256_storeu_pd(BO + 0, t1);
			_mm256_storeu_pd(BO + 4, t2);
			_mm256_storeu_pd(BO + 8, t3);

			BO1 += 8;
			BO2 += 8;
			BO  += 12;
			i--;
		} while (i != 0);


		CO1 = C;
		C += ldc * 12;

		AO = A + 16;
		i = M / 4;

		while (i > 0) {
			// L12_11
			__m256d ymm0, ymm1, ymm2, ymm3;
			__m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
			int Kdiv8 = K / 8;
			int Kmod8 = K & 7;
			BO = BUFFER1;
			BO += 12;
	
			if (Kdiv8 >=2) {
			
				KERNEL4x12_I()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				Kdiv8 -= 2;
				while (Kdiv8 > 0) {
					//L12_12
					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()

					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()
					Kdiv8--;
				}
				// L12_12a
				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_E()
				
			} else {
				// L12_13
				if (Kdiv8 == 1) {
					KERNEL4x12_I()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()

					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_E()
				} else {
					// L12_14
					INIT4x12()
				}
			}	
			// L12_16
			while (Kmod8 > 0) {
				// L12_17
				KERNEL4x12_SUB()
				Kmod8--;
			}
			// L12_19
			SAVE4x12(alpha)

			i--;

		} 
/**************************************************************************
* Rest of M 
***************************************************************************/
		// L12_20
		if (M & 3) {
			if (M & 2) {
				__m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
				BO = BUFFER1;
				BO += 12;

				INIT2x12()
				i = K;
				
				while (i >= 8) {
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()

					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					i -= 8;
				}
				while (i > 0) {
					KERNEL2x12_SUB()
					i--;
				}
				SAVE2x12(alpha)

			}
			// L12_40
			if (M & 1) {
				double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8, dbl9, dbl10, dbl11, dbl12, dbl13, dbl14, dbl15;
				BO = BUFFER1;
				BO += 12;
				INIT1x12()
				
				i = K;
				
				while (i >= 8) {
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()

					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					i -= 8;
				}
				while (i > 0) {
					KERNEL1x12_SUB()
					i--;
				}
				SAVE1x12(alpha)

			}
		}
		// L12_100


/**************************************************************************************************/
		// L13_01
		BO2 = B;
		BO3 = B + K * 8;
		B = BO3 + K * 8;
		BO = BUFFER1;
		i = K;
		do {
			// L13_02b
			__m256d t1, t2, t3;

			t1 = _mm256_loadu_pd(BO2+4);
			t2 = _mm256_loadu_pd(BO3);
			t3 = _mm256_loadu_pd(BO3+4);

			_mm256_storeu_pd(BO + 0, t1);
			_mm256_storeu_pd(BO + 4, t2);
			_mm256_storeu_pd(BO + 8, t3);

			BO2 += 8;
			BO3 += 8;
			BO  += 12;
			i--;
		} while (i != 0);


		// L13_10
		CO1 = C;
		C += 12 * ldc;

		AO = A + 16;
		i = m /4;

		while (i > 0) {
			__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
			// L13_11
			int Kdiv8 = K / 8;
			int Kmod8 = K & 7;
			BO = BUFFER1;
			BO += 12;
	
			if (Kdiv8 >=2) {
			
				KERNEL4x12_I()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				Kdiv8 -= 2;
				while (Kdiv8 > 0) {
					//L13_12
					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()

					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()
					Kdiv8--;
				}
				// L13_12a
				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_M2()

				KERNEL4x12_M1()
				KERNEL4x12_M2()
				KERNEL4x12_M1()
				KERNEL4x12_E()
				
			} else {
				// L13_13
				if (Kdiv8 == 1) {
					KERNEL4x12_I()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_M2()

					KERNEL4x12_M1()
					KERNEL4x12_M2()
					KERNEL4x12_M1()
					KERNEL4x12_E()
				} else {
					// L12_14
					INIT4x12()
				}
			}	
			// L12_16
			while (Kmod8 > 0) {
				// L12_17
				KERNEL4x12_SUB()
				Kmod8--;
			}
			// L12_19
			SAVE4x12(alpha)

			i--;
			


		}
		// L13_20


/**************************************************************************
* Rest of M 
***************************************************************************/
		if (M & 3) {
			if (M & 2) {
				__m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
				BO = BUFFER1;
				BO += 12;

				INIT2x12()
				i = K;
				
				while (i >= 8) {
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()

					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					KERNEL2x12_SUB()
					i -= 8;
				}
				while (i > 0) {
					KERNEL2x12_SUB()
					i--;
				}
				SAVE2x12(alpha)

			}
			// L13_40
			if (M & 1) {
				double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8, dbl9, dbl10, dbl11, dbl12, dbl13, dbl14, dbl15;
				BO = BUFFER1;
				BO += 12;
				INIT1x12()
				
				i = K;
				
				while (i >= 8) {
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()

					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					KERNEL1x12_SUB()
					i -= 8;
				}
				while (i > 0) {
					KERNEL1x12_SUB()
					i--;
				}
				SAVE1x12(alpha)

			}
		}
		// L13_100
		J--;
	}	
/**************************************************************************************************/

	if (Nmod24 == 0)
		return 0;   /* we're done */

		// L8_0
	while (Nmod24 >= 8) {
		double *BO;
		double *CO1;
		double *AO;
		int i;
			// L8_10
			CO1 = C;
			C += 8 * ldc;

			AO = A + 16;

			i = m/4;
			while (i > 0) {
				__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
				// L8_11
				BO = B + 12;
				int Kdiv8 = K / 8;
				int Kmod8 = K & 7;
	
				if (Kdiv8 >=2) {
			
					KERNEL4x8_I()
					KERNEL4x8_M2()
					KERNEL4x8_M1()
					KERNEL4x8_M2()
	
					KERNEL4x8_M1()
					KERNEL4x8_M2()
					KERNEL4x8_M1()
					KERNEL4x8_M2()

					Kdiv8 -= 2;
					while (Kdiv8 > 0) {
						//L8_12
						KERNEL4x8_M1()
						KERNEL4x8_M2()
						KERNEL4x8_M1()
						KERNEL4x8_M2()
	
						KERNEL4x8_M1()
						KERNEL4x8_M2()
						KERNEL4x8_M1()
						KERNEL4x8_M2()
						Kdiv8--;
					}
					// L8_12a
					KERNEL4x8_M1()
					KERNEL4x8_M2()
					KERNEL4x8_M1()
					KERNEL4x8_M2()
	
					KERNEL4x8_M1()
					KERNEL4x8_M2()
					KERNEL4x8_M1()
					KERNEL4x8_E()
					
				} else {
					// L8_13
					if (Kdiv8 == 1) {
						KERNEL4x8_I()
						KERNEL4x8_M2()
						KERNEL4x8_M1()
						KERNEL4x8_M2()
	
						KERNEL4x8_M1()
						KERNEL4x8_M2()
						KERNEL4x8_M1()
						KERNEL4x8_E()
					} else {
						// L8_14
						INIT4x8()
					}
				}	
				// L8_16
				while (Kmod8 > 0) {
					// L12_17
					KERNEL4x8_SUB()
					Kmod8--;
				}
				// L8_19
				SAVE4x8(alpha)
	
				i--;
			}

/**************************************************************************
* Rest of M 
***************************************************************************/

			if (M & 3) {
				if (M & 2) {
					__m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11;
					BO = B;
					BO += 12;

					INIT2x8()
					i = K;
				
					while (i >= 8) {
						KERNEL2x8_SUB()
						KERNEL2x8_SUB()
						KERNEL2x8_SUB()
						KERNEL2x8_SUB()

						KERNEL2x8_SUB()
						KERNEL2x8_SUB()
						KERNEL2x8_SUB()
						KERNEL2x8_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL2x8_SUB()
						i--;
					}
					SAVE2x8(alpha)

				}
				// L13_40
				if (M & 1) {
					double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8, dbl9, dbl10, dbl11;
					BO = B;
					BO += 12;
					INIT1x8()
					
					i = K;
					
					while (i >= 8) {
						KERNEL1x8_SUB()
						KERNEL1x8_SUB()
						KERNEL1x8_SUB()
						KERNEL1x8_SUB()

						KERNEL1x8_SUB()
						KERNEL1x8_SUB()
						KERNEL1x8_SUB()
						KERNEL1x8_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL1x8_SUB()
						i--;
					}
					SAVE1x8(alpha)

				}
			}
		B += K * 8;
		Nmod24 -= 8;
	}

	if (Nmod24 == 0)
		return 0;	
	

		// L8_0
	while (Nmod24 >= 4) {
		double *BO;
		double *CO1;
		double *AO;
		int i;
			// L8_10
			CO1 = C;
			C += 4 * ldc;

			AO = A + 16;

			i = m/4;
			while (i > 0) {
				// L8_11
				__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
				BO = B + 12;
				int Kdiv8 = K / 8;
				int Kmod8 = K & 7;
	
				if (Kdiv8 >=2) {
			
					KERNEL4x4_I()
					KERNEL4x4_M2()
					KERNEL4x4_M1()
					KERNEL4x4_M2()
	
					KERNEL4x4_M1()
					KERNEL4x4_M2()
					KERNEL4x4_M1()
					KERNEL4x4_M2()

					Kdiv8 -= 2;
					while (Kdiv8 > 0) {
						//L8_12
						KERNEL4x4_M1()
						KERNEL4x4_M2()
						KERNEL4x4_M1()
						KERNEL4x4_M2()
	
						KERNEL4x4_M1()
						KERNEL4x4_M2()
						KERNEL4x4_M1()
						KERNEL4x4_M2()
						Kdiv8--;
					}
					// L8_12a
					KERNEL4x4_M1()
					KERNEL4x4_M2()
					KERNEL4x4_M1()
					KERNEL4x4_M2()
	
					KERNEL4x4_M1()
					KERNEL4x4_M2()
					KERNEL4x4_M1()
					KERNEL4x4_E()
					
				} else {
					// L8_13
					if (Kdiv8 == 1) {
						KERNEL4x4_I()
						KERNEL4x4_M2()
						KERNEL4x4_M1()
						KERNEL4x4_M2()
	
						KERNEL4x4_M1()
						KERNEL4x4_M2()
						KERNEL4x4_M1()
						KERNEL4x4_E()
					} else {
						// L8_14
						INIT4x4()
					}
				}	
				// L8_16
				while (Kmod8 > 0) {
					// L12_17
					KERNEL4x4_SUB()
					Kmod8--;
				}
				// L8_19
				SAVE4x4(alpha)
	
				i--;
			}

/**************************************************************************
* Rest of M 
***************************************************************************/

			if (M & 3) {
				if (M & 2) {
					__m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
					BO = B;
					BO += 12;

					INIT2x4()
					i = K;
				
					while (i >= 8) {
						KERNEL2x4_SUB()
						KERNEL2x4_SUB()
						KERNEL2x4_SUB()
						KERNEL2x4_SUB()

						KERNEL2x4_SUB()
						KERNEL2x4_SUB()
						KERNEL2x4_SUB()
						KERNEL2x4_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL2x4_SUB()
						i--;
					}
					SAVE2x4(alpha)

				}
				// L13_40
				if (M & 1) {
					double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8;
					BO = B;
					BO += 12;
					INIT1x4()
					
					i = K;
					
					while (i >= 8) {
						KERNEL1x4_SUB()
						KERNEL1x4_SUB()
						KERNEL1x4_SUB()
						KERNEL1x4_SUB()

						KERNEL1x4_SUB()
						KERNEL1x4_SUB()
						KERNEL1x4_SUB()
						KERNEL1x4_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL1x4_SUB()
						i--;
					}
					SAVE1x4(alpha)

				}
			}
		B += K * 4;
		Nmod24 -= 4;
	}

/**************************************************************************************************/

		// L8_0
	while (Nmod24 >= 2) {
		double *BO;
		double *CO1;
		double *AO;
		int i;
			// L8_10
			CO1 = C;
			C += 2 * ldc;

			AO = A + 16;

			i = m/4;
			while (i > 0) {
				__m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
				// L8_11
				BO = B + 12;
				int Kmod8 = K;
	
				INIT4x2()

				// L8_16
				while (Kmod8 > 0) {
					// L12_17
					KERNEL4x2_SUB()
					Kmod8--;
				}
				// L8_19
				SAVE4x2(alpha)
	
				i--;
			}

/**************************************************************************
* Rest of M 
***************************************************************************/

			if (M & 3) {
				if (M & 2) {
					__m128d xmm0, xmm2, xmm3, xmm4, xmm6;
					BO = B;
					BO += 12;

					INIT2x2()
					i = K;
				
					while (i >= 8) {
						KERNEL2x2_SUB()
						KERNEL2x2_SUB()
						KERNEL2x2_SUB()
						KERNEL2x2_SUB()

						KERNEL2x2_SUB()
						KERNEL2x2_SUB()
						KERNEL2x2_SUB()
						KERNEL2x2_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL2x2_SUB()
						i--;
					}
					SAVE2x2(alpha)

				}
				// L13_40
				if (M & 1) {
					double dbl0, dbl1, dbl2, dbl4, dbl5;
					BO = B;
					BO += 12;
					INIT1x2()
					
					i = K;
					
					while (i >= 8) {
						KERNEL1x2_SUB()
						KERNEL1x2_SUB()
						KERNEL1x2_SUB()
						KERNEL1x2_SUB()

						KERNEL1x2_SUB()
						KERNEL1x2_SUB()
						KERNEL1x2_SUB()
						KERNEL1x2_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL1x2_SUB()
						i--;
					}
					SAVE1x2(alpha)

				}
			}
		B += K * 2;
		Nmod24 -= 2;
	}

		// L8_0
	while (Nmod24 >= 1) {
			// L8_10
		double *BO;
		double *CO1;
		double *AO;
		int i;
			CO1 = C;
			C += ldc;

			AO = A + 16;

			i = m/4;
			while (i > 0) {
				__m256d ymm0, ymm2, ymm4, ymm5, ymm6, ymm7;
				// L8_11
				BO = B + 12;
				int Kmod8 = K;
	
				INIT4x1()
				// L8_16
				while (Kmod8 > 0) {
					// L12_17
					KERNEL4x1_SUB()
					Kmod8--;
				}
				// L8_19
				SAVE4x1(alpha)
	
				i--;
			}

/**************************************************************************
* Rest of M 
***************************************************************************/

			if (M & 3) {
				if (M & 2) {
					__m128d xmm0, xmm2, xmm4;
					BO = B;
					BO += 12;

					INIT2x1()
					i = K;
				
					while (i >= 8) {
						KERNEL2x1_SUB()
						KERNEL2x1_SUB()
						KERNEL2x1_SUB()
						KERNEL2x1_SUB()

						KERNEL2x1_SUB()
						KERNEL2x1_SUB()
						KERNEL2x1_SUB()
						KERNEL2x1_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL2x1_SUB()
						i--;
					}
					SAVE2x1(alpha)

				}
				// L13_40
				if (M & 1) {
					double dbl0, dbl1, dbl4;
					BO = B;
					BO += 12;
					INIT1x1()
					
					i = K;
					
					while (i >= 8) {
						KERNEL1x1_SUB()
						KERNEL1x1_SUB()
						KERNEL1x1_SUB()
						KERNEL1x1_SUB()

						KERNEL1x1_SUB()
						KERNEL1x1_SUB()
						KERNEL1x1_SUB()
						KERNEL1x1_SUB()
						i -= 8;
					}
					while (i > 0) {
						KERNEL1x1_SUB()
						i--;
					}
					SAVE1x1(alpha)

				}
			}
		B += K * 1;
		Nmod24 -= 1;
	}


	return 0;
}