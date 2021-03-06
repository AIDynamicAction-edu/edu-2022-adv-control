/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) test_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[9] = {1, 3, 0, 1, 2, 3, 0, 0, 0};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};

/* rhs:(i0[1x3],i1[3],i2[3])->(o0[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a5, a6, a7, a8, a9;
  a0=1.0129999999999999e+00;
  a1=2.7326000000000000e-02;
  a2=arg[0]? arg[0][0] : 0;
  a3=sin(a2);
  a3=(a1*a3);
  a4=arg[0]? arg[0][1] : 0;
  a5=sin(a4);
  a3=(a3*a5);
  a5=(a0*a3);
  a6=(a5*a3);
  a7=-2.7326000000000000e-02;
  a8=cos(a4);
  a8=(a7*a8);
  a9=(a0*a8);
  a9=(a9*a8);
  a6=(a6+a9);
  a9=sin(a4);
  a9=(a1*a9);
  a10=cos(a2);
  a9=(a9*a10);
  a10=(a0*a9);
  a11=(a10*a9);
  a6=(a6+a11);
  a11=1.7199999999999999e-01;
  a12=1.3197000000000000e-01;
  a13=arg[0]? arg[0][2] : 0;
  a14=(a13+a4);
  a14=sin(a14);
  a14=(a12*a14);
  a15=sin(a2);
  a14=(a14*a15);
  a15=2.0000000000000001e-01;
  a16=sin(a2);
  a16=(a15*a16);
  a17=sin(a4);
  a16=(a16*a17);
  a14=(a14+a16);
  a16=(a11*a14);
  a17=(a16*a14);
  a18=-1.3197000000000000e-01;
  a19=(a13+a4);
  a19=cos(a19);
  a19=(a18*a19);
  a20=cos(a4);
  a20=(a15*a20);
  a19=(a19-a20);
  a20=(a11*a19);
  a21=(a20*a19);
  a17=(a17+a21);
  a21=(a13+a4);
  a21=sin(a21);
  a21=(a12*a21);
  a22=cos(a2);
  a21=(a21*a22);
  a22=sin(a4);
  a22=(a15*a22);
  a23=cos(a2);
  a22=(a22*a23);
  a21=(a21+a22);
  a22=(a11*a21);
  a23=(a22*a21);
  a17=(a17+a23);
  a6=(a6+a17);
  a17=(a13+a4);
  a17=sin(a17);
  a17=(a12*a17);
  a23=sin(a2);
  a17=(a17*a23);
  a23=(a11*a17);
  a24=(a23*a17);
  a25=(a13+a4);
  a25=cos(a25);
  a18=(a18*a25);
  a25=(a11*a18);
  a26=(a25*a18);
  a24=(a24+a26);
  a26=(a13+a4);
  a26=sin(a26);
  a26=(a12*a26);
  a27=cos(a2);
  a26=(a26*a27);
  a27=(a11*a26);
  a28=(a27*a26);
  a24=(a24+a28);
  a28=(a6*a24);
  a29=(a16*a17);
  a20=(a20*a18);
  a29=(a29+a20);
  a20=(a22*a26);
  a29=(a29+a20);
  a20=(a23*a14);
  a25=(a25*a19);
  a20=(a20+a25);
  a25=(a27*a21);
  a20=(a20+a25);
  a25=(a29*a20);
  a28=(a28-a25);
  a25=-6.1473000000000000e-02;
  a30=sin(a2);
  a30=(a25*a30);
  a31=cos(a2);
  a31=(a1*a31);
  a32=cos(a4);
  a31=(a31*a32);
  a30=(a30-a31);
  a31=(a0*a30);
  a32=(a31*a3);
  a33=cos(a2);
  a33=(a25*a33);
  a34=sin(a2);
  a34=(a1*a34);
  a35=cos(a4);
  a34=(a34*a35);
  a33=(a33+a34);
  a34=(a0*a33);
  a35=(a34*a9);
  a32=(a32+a35);
  a35=-8.3799999999999999e-02;
  a36=sin(a2);
  a36=(a35*a36);
  a37=(a13+a4);
  a37=cos(a37);
  a37=(a12*a37);
  a38=cos(a2);
  a37=(a37*a38);
  a36=(a36-a37);
  a37=cos(a2);
  a37=(a15*a37);
  a38=cos(a4);
  a37=(a37*a38);
  a36=(a36-a37);
  a37=(a11*a36);
  a38=(a37*a14);
  a39=cos(a2);
  a39=(a35*a39);
  a40=sin(a2);
  a40=(a12*a40);
  a41=(a13+a4);
  a41=cos(a41);
  a40=(a40*a41);
  a39=(a39+a40);
  a40=sin(a2);
  a40=(a15*a40);
  a41=cos(a4);
  a40=(a40*a41);
  a39=(a39+a40);
  a40=(a11*a39);
  a41=(a40*a21);
  a38=(a38+a41);
  a32=(a32+a38);
  a5=(a5*a30);
  a10=(a10*a33);
  a5=(a5+a10);
  a16=(a16*a36);
  a22=(a22*a39);
  a16=(a16+a22);
  a5=(a5+a16);
  a16=(a5*a24);
  a23=(a23*a36);
  a27=(a27*a39);
  a23=(a23+a27);
  a27=(a29*a23);
  a16=(a16-a27);
  a16=(a32*a16);
  a31=(a31*a30);
  a34=(a34*a33);
  a31=(a31+a34);
  a34=(a37*a36);
  a27=(a40*a39);
  a34=(a34+a27);
  a31=(a31+a34);
  a34=(a6*a24);
  a27=(a29*a20);
  a34=(a34-a27);
  a34=(a31*a34);
  a16=(a16-a34);
  a37=(a37*a17);
  a40=(a40*a26);
  a37=(a37+a40);
  a40=(a5*a20);
  a34=(a6*a23);
  a40=(a40-a34);
  a40=(a37*a40);
  a16=(a16-a40);
  a28=(a28/a16);
  a30=(a0*a30);
  a40=cos(a2);
  a25=(a25*a40);
  a40=arg[1]? arg[1][0] : 0;
  a25=(a25*a40);
  a34=sin(a2);
  a34=(a1*a34);
  a27=cos(a4);
  a34=(a34*a27);
  a34=(a34*a40);
  a25=(a25+a34);
  a34=sin(a4);
  a34=(a1*a34);
  a27=cos(a2);
  a34=(a34*a27);
  a27=arg[1]? arg[1][1] : 0;
  a34=(a34*a27);
  a25=(a25+a34);
  a34=(a30*a25);
  a22=(a0*a33);
  a10=6.1473000000000000e-02;
  a38=sin(a2);
  a10=(a10*a38);
  a10=(a10*a40);
  a38=sin(a2);
  a38=(a1*a38);
  a41=sin(a4);
  a38=(a38*a41);
  a38=(a38*a27);
  a10=(a10-a38);
  a38=cos(a2);
  a38=(a1*a38);
  a41=cos(a4);
  a38=(a38*a41);
  a38=(a38*a40);
  a10=(a10+a38);
  a38=(a22*a10);
  a34=(a34+a38);
  a36=(a11*a36);
  a38=cos(a2);
  a35=(a35*a38);
  a35=(a35*a40);
  a38=arg[1]? arg[1][2] : 0;
  a41=(a38+a27);
  a41=(a12*a41);
  a42=(a13+a4);
  a42=sin(a42);
  a41=(a41*a42);
  a42=cos(a2);
  a41=(a41*a42);
  a35=(a35+a41);
  a41=sin(a2);
  a41=(a12*a41);
  a42=(a13+a4);
  a42=cos(a42);
  a41=(a41*a42);
  a41=(a41*a40);
  a35=(a35+a41);
  a41=sin(a2);
  a41=(a15*a41);
  a42=cos(a4);
  a41=(a41*a42);
  a41=(a41*a40);
  a35=(a35+a41);
  a41=sin(a4);
  a41=(a15*a41);
  a42=cos(a2);
  a41=(a41*a42);
  a41=(a41*a27);
  a35=(a35+a41);
  a41=(a36*a35);
  a42=(a11*a39);
  a43=8.3799999999999999e-02;
  a44=sin(a2);
  a43=(a43*a44);
  a43=(a43*a40);
  a44=(a38+a27);
  a44=(a12*a44);
  a45=(a13+a4);
  a45=sin(a45);
  a44=(a44*a45);
  a45=sin(a2);
  a44=(a44*a45);
  a43=(a43-a44);
  a44=(a13+a4);
  a44=cos(a44);
  a44=(a12*a44);
  a45=cos(a2);
  a44=(a44*a45);
  a44=(a44*a40);
  a43=(a43+a44);
  a44=sin(a2);
  a44=(a15*a44);
  a45=sin(a4);
  a44=(a44*a45);
  a44=(a44*a27);
  a43=(a43-a44);
  a44=cos(a2);
  a44=(a15*a44);
  a45=cos(a4);
  a44=(a44*a45);
  a44=(a44*a40);
  a43=(a43+a44);
  a44=(a42*a43);
  a41=(a41+a44);
  a34=(a34+a41);
  a34=(a34*a40);
  a41=sin(a2);
  a41=(a1*a41);
  a44=cos(a4);
  a41=(a41*a44);
  a41=(a41*a27);
  a44=sin(a4);
  a44=(a1*a44);
  a45=cos(a2);
  a44=(a44*a45);
  a44=(a44*a40);
  a41=(a41+a44);
  a30=(a30*a41);
  a44=sin(a2);
  a7=(a7*a44);
  a44=sin(a4);
  a7=(a7*a44);
  a7=(a7*a40);
  a44=cos(a2);
  a44=(a1*a44);
  a45=cos(a4);
  a44=(a44*a45);
  a44=(a44*a27);
  a7=(a7+a44);
  a22=(a22*a7);
  a30=(a30+a22);
  a22=(a38+a27);
  a22=(a12*a22);
  a44=sin(a2);
  a22=(a22*a44);
  a44=(a13+a4);
  a44=cos(a44);
  a22=(a22*a44);
  a44=(a13+a4);
  a44=sin(a44);
  a44=(a12*a44);
  a45=cos(a2);
  a44=(a44*a45);
  a44=(a44*a40);
  a22=(a22+a44);
  a44=sin(a2);
  a44=(a15*a44);
  a45=cos(a4);
  a44=(a44*a45);
  a44=(a44*a27);
  a22=(a22+a44);
  a44=sin(a4);
  a44=(a15*a44);
  a45=cos(a2);
  a44=(a44*a45);
  a44=(a44*a40);
  a22=(a22+a44);
  a44=(a36*a22);
  a45=(a38+a27);
  a45=(a12*a45);
  a46=(a13+a4);
  a46=cos(a46);
  a45=(a45*a46);
  a46=cos(a2);
  a45=(a45*a46);
  a46=(a13+a4);
  a46=sin(a46);
  a46=(a12*a46);
  a47=sin(a2);
  a46=(a46*a47);
  a46=(a46*a40);
  a45=(a45-a46);
  a46=sin(a2);
  a46=(a15*a46);
  a47=sin(a4);
  a46=(a46*a47);
  a46=(a46*a40);
  a45=(a45-a46);
  a46=cos(a2);
  a46=(a15*a46);
  a47=cos(a4);
  a46=(a46*a47);
  a46=(a46*a27);
  a45=(a45+a46);
  a46=(a42*a45);
  a44=(a44+a46);
  a30=(a30+a44);
  a30=(a30*a27);
  a34=(a34+a30);
  a30=(a38+a27);
  a30=(a12*a30);
  a44=sin(a2);
  a30=(a30*a44);
  a44=(a13+a4);
  a44=cos(a44);
  a30=(a30*a44);
  a44=(a13+a4);
  a44=sin(a44);
  a44=(a12*a44);
  a46=cos(a2);
  a44=(a44*a46);
  a44=(a44*a40);
  a30=(a30+a44);
  a36=(a36*a30);
  a44=(a38+a27);
  a44=(a12*a44);
  a46=(a13+a4);
  a46=cos(a46);
  a44=(a44*a46);
  a46=cos(a2);
  a44=(a44*a46);
  a46=(a13+a4);
  a46=sin(a46);
  a46=(a12*a46);
  a2=sin(a2);
  a46=(a46*a2);
  a46=(a46*a40);
  a44=(a44-a46);
  a42=(a42*a44);
  a36=(a36+a42);
  a36=(a36*a38);
  a34=(a34+a36);
  a36=9.8000000000000007e+00;
  a33=(a0*a33);
  a39=(a11*a39);
  a33=(a33+a39);
  a33=(a36*a33);
  a34=(a34+a33);
  a33=arg[2]? arg[2][0] : 0;
  a34=(a34-a33);
  a28=(a28*a34);
  a33=(a32*a24);
  a39=(a37*a20);
  a33=(a33-a39);
  a33=(a33/a16);
  a3=(a0*a3);
  a25=(a3*a25);
  a39=(a0*a9);
  a10=(a39*a10);
  a25=(a25+a10);
  a14=(a11*a14);
  a10=(a14*a35);
  a42=(a11*a21);
  a46=(a42*a43);
  a10=(a10+a46);
  a25=(a25+a10);
  a25=(a25*a40);
  a3=(a3*a41);
  a8=(a0*a8);
  a41=sin(a4);
  a1=(a1*a41);
  a1=(a1*a27);
  a8=(a8*a1);
  a3=(a3+a8);
  a39=(a39*a7);
  a3=(a3+a39);
  a39=(a14*a22);
  a19=(a11*a19);
  a7=(a38+a27);
  a7=(a12*a7);
  a8=(a13+a4);
  a8=sin(a8);
  a7=(a7*a8);
  a8=sin(a4);
  a15=(a15*a8);
  a15=(a15*a27);
  a7=(a7+a15);
  a15=(a19*a7);
  a39=(a39+a15);
  a15=(a42*a45);
  a39=(a39+a15);
  a3=(a3+a39);
  a3=(a3*a27);
  a25=(a25+a3);
  a14=(a14*a30);
  a3=(a38+a27);
  a12=(a12*a3);
  a13=(a13+a4);
  a13=sin(a13);
  a12=(a12*a13);
  a19=(a19*a12);
  a14=(a14+a19);
  a42=(a42*a44);
  a14=(a14+a42);
  a14=(a14*a38);
  a25=(a25+a14);
  a0=(a0*a9);
  a21=(a11*a21);
  a0=(a0+a21);
  a0=(a36*a0);
  a25=(a25+a0);
  a0=arg[2]? arg[2][1] : 0;
  a25=(a25-a0);
  a33=(a33*a25);
  a28=(a28-a33);
  a33=(a32*a29);
  a0=(a37*a6);
  a33=(a33-a0);
  a33=(a33/a16);
  a17=(a11*a17);
  a35=(a17*a35);
  a0=(a11*a26);
  a43=(a0*a43);
  a35=(a35+a43);
  a35=(a35*a40);
  a22=(a17*a22);
  a18=(a11*a18);
  a7=(a18*a7);
  a22=(a22+a7);
  a45=(a0*a45);
  a22=(a22+a45);
  a22=(a22*a27);
  a35=(a35+a22);
  a17=(a17*a30);
  a18=(a18*a12);
  a17=(a17+a18);
  a0=(a0*a44);
  a17=(a17+a0);
  a17=(a17*a38);
  a35=(a35+a17);
  a11=(a11*a26);
  a36=(a36*a11);
  a35=(a35+a36);
  a36=arg[2]? arg[2][2] : 0;
  a35=(a35-a36);
  a33=(a33*a35);
  a28=(a28+a33);
  if (res[0]!=0) res[0][0]=a28;
  a28=(a31*a24);
  a33=(a37*a23);
  a28=(a28-a33);
  a28=(a28/a16);
  a28=(a28*a25);
  a24=(a5*a24);
  a33=(a29*a23);
  a24=(a24-a33);
  a24=(a24/a16);
  a24=(a24*a34);
  a28=(a28-a24);
  a29=(a31*a29);
  a37=(a37*a5);
  a29=(a29-a37);
  a29=(a29/a16);
  a29=(a29*a35);
  a28=(a28-a29);
  if (res[0]!=0) res[0][1]=a28;
  a28=(a5*a20);
  a29=(a6*a23);
  a28=(a28-a29);
  a28=(a28/a16);
  a28=(a28*a34);
  a20=(a31*a20);
  a23=(a32*a23);
  a20=(a20-a23);
  a20=(a20/a16);
  a20=(a20*a25);
  a28=(a28-a20);
  a31=(a31*a6);
  a32=(a32*a5);
  a31=(a31-a32);
  a31=(a31/a16);
  a31=(a31*a35);
  a28=(a28+a31);
  if (res[0]!=0) res[0][2]=a28;
  return 0;
}

CASADI_SYMBOL_EXPORT int rhs(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int rhs_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int rhs_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rhs_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int rhs_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rhs_release(int mem) {
}

CASADI_SYMBOL_EXPORT void rhs_incref(void) {
}

CASADI_SYMBOL_EXPORT void rhs_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int rhs_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int rhs_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real rhs_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rhs_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rhs_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rhs_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rhs_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int rhs_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
