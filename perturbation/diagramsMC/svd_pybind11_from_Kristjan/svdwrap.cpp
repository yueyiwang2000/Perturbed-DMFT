#include <cstdint>
#include <ostream>
#include <deque>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pystreambuf.h"
#include "svd_func.h"
namespace py = pybind11;

class ConvertArray{
  py::array_t<double, py::array::c_style> A;
public:
  ConvertArray(const bl::Array<double,1>& _A_) : A(  _A_.extent(0)  )
  {
    auto q = A.mutable_unchecked<1>();
    for (int i = 0; i < q.shape(0); i++) q(i) = _A_(i);
  }
  py::array_t<double> operator()(){return A;}
};
//void inverse_fourier_fermion(bl::Array<double,1>& Gtau, double beta, const bl::Array<double,1>& tau, int nom, const bl::Array<int,1>& iom, const bl::Array<complex<double>,1>& Giom, const bl::TinyVector<double,2>& ah)

py::array_t<double> InverseFourierFermion(double beta, py::array_t<double>& _tau_, int nom, py::array_t<int>& _iom_, py::array_t<std::complex<double>>& _Giom_, py::array_t<double>& _ah_)
{ // nom : number of points before the tail.
  // At high frequency we subtract G_approx ~ ah(ik,0)/(iom-ah(ik,1)) and treat it analytically
  //
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  
  py::buffer_info info_Gw = _Giom_.request();
  if (info_Gw.ndim != 2) throw std::runtime_error("Number of dimensions for Giom should be 2.");
  int Nq = info_Gw.shape[0];
  int Nw = info_Gw.shape[1];
  int Nt = tau.extent(0);

  py::array_t<double,py::array::c_style> _Gt_({ Nq, Nt });
  py::buffer_info info_Gt = _Gt_.request();
  
  auto ahs = _ah_.mutable_unchecked<2>();
  for (int iq=0; iq<Nq; iq++){ // over all q-points
    bl::Array<std::complex<double>,1> G_iom( (std::complex<double>*)info_Gw.ptr + iq*Nw, Nw, bl::neverDeleteData);
    bl::Array<double,1> G_tau( (double*)info_Gt.ptr + iq*Nt, Nt, bl::neverDeleteData);
    bl::TinyVector<double,2> ah( ahs(iq,0), ahs(iq,1) );
    //cout << "ah["<<iq<<"]=" << ah <<  endl;
    inverse_fourier_fermion(G_tau, beta, tau, nom, iom, G_iom, ah);
  }
  return _Gt_;
}

py::array_t<std::complex<double>> FourierFermion(double beta, py::array_t<int>& _iom_, py::array_t<double>& _tau_, py::array_t<double>& Gtau)
{
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  py::buffer_info info_Gt = Gtau.request();
  if (info_Gt.ndim != 2) throw std::runtime_error("Number of dimensions for Gt should be 2.");
  bl::Array<double,2> Gt( (double*)info_Gt.ptr, bl::shape(info_Gt.shape[0],info_Gt.shape[1]), bl::neverDeleteData);

  py::array_t<std::complex<double>, py::array::c_style> _Gqw_({ Gt.extent(0), iom.extent(0) });
  auto Gqw = _Gqw_.mutable_unchecked<2>();
 
  int Nt = tau.extent(0);
  for (int iq=0; iq<Gt.extent(0); iq++){
    Spline1D<double> wt(Nt);
    for (int it=0; it<Nt; it++) wt[it] = Gt(iq,it);
    wt.splineIt(tau);
    for (int in=0; in<iom.extent(0); in++){
      double om = (2*iom(in)+1)*M_PI/beta;
      Gqw(iq,in) = wt.Fourier(om, tau);
    }
  }
  return _Gqw_;
}

PYBIND11_MODULE(svdwrap,m) {
  m.doc() = "pybind11 wrap for svd-functions";

  py::class_<intpar> intpar(m, "intpar", "small class used for interpolation with mesh1D and spline1D");
  intpar.def(py::init<int,double>())
    .def_readwrite("i", &intpar::i, "The integer index on the mesh")
    .def_readwrite("p", &intpar::p, "The distance between this and next point p=(x-x_i)/(x_{i+1}-x_i)")
    ;

  m.def("InverseFourierFermion", &InverseFourierFermion, "Inverse Fourier Transform for fermionic set of functions");
  m.def("FourierFermion", &FourierFermion, "Fourier transform for fermionic set of functions");
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //             SVDFunc
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  py::class_<SVDFunc> SVDFunc(m, "SVDFunc", "Contains SVD obtained basis functions for imaginary time quantities");
  SVDFunc.def(py::init<>())
    .def("cmp", &SVDFunc::cmp, "Computes SVD functions",
	 py::arg("statistics"),py::arg("beta"),py::arg("lmax"),py::arg("Ntau"),py::arg("x0"),py::arg("L"),py::arg("Nw"),py::arg("clog"),py::arg("Print")=true,py::arg("k_ovlp")=5)
    .def("CheckOverlap", &SVDFunc::CheckOverlap, "Check if the SVD functions are fine.")
    //.def("Compute_u_bose_iom", &SVDFunc::Compute_u_bose_iom, "Computes u_l(iom) for bosonic functions.")
    .def("Read", &SVDFunc::Read, "Reads ul(tau) functions from file, and possibly adds some extra", py::arg("filename"), py::arg("statistics"), py::arg("lmax_")=0, py::arg("k_ovlp_")=5)
    .def("get_lll", &SVDFunc::get_lll, "From integer index computes three ls, namely bose-l, fermi-l, fermi-l.", py::arg("ii"),py::arg("lmax_")=0)
    .def_readonly("om", &SVDFunc::om, "The real axis mesh")
    .def_readonly("k_ovlp", &SVDFunc::k_ovlp, "Number of points for romberg interpolation")
    //.def_readonly("fU_ferm", &SVDFunc::fU, "SVD basis functions in time fU_ferm[l][it]")
    //.def_readonly("fU_bose", &SVDFunc::fU_bose, "SVD basis functions in time fU_bose[l][it]")
    .def_readonly("lmax", &SVDFunc::lmax, "l cutoff for fermi functions")
    .def_readonly("lmax_bose", &SVDFunc::lmax_bose, "l cutoff for bose functions")
    .def("tau", [](class SVDFunc& svd) ->py::array_t<double>{
	ConvertArray ca(svd.tau);
	return ca();
      })
    .def("fU_ferm", [](class SVDFunc& svd) ->py::array_t<double>{
	py::array_t<double, py::array::c_style> fUp({static_cast<int>(svd.fU.size()),svd.fU[0].f.extent(0)});
	auto r = fUp.mutable_unchecked<2>();
	for (int l = 0; l < r.shape(0); l++)
	  for (int it = 0; it < r.shape(1); it++)
	    r(l,it) = svd.fU[l].f(it);
	return fUp;
      })
    .def("MatsubaraFrequency", [](class SVDFunc& svd, py::array_t<double>& _g_, py::array_t<int>& _iOm_, const string& statistics="fermi")->py::array_t<complex<double>>{
	//void MatsubaraFrequency(bl::Array<complex<double>,1>& giom, const bl::Array<double,1>& g, const bl::Array<int,1> iw, const string& statistics="fermi"){
	py::buffer_info info_w = _iOm_.request();
	bl::Array<int,1> iOm((int*)info_w.ptr, info_w.shape[0], bl::neverDeleteData);
	py::buffer_info info_g = _g_.request();
	bl::Array<double,1> g((double*)info_g.ptr, info_g.shape[0], bl::neverDeleteData);
	
	py::array_t<complex<double>> _giom_( iOm.size() );
	py::buffer_info info_N = _giom_.request();
	bl::Array<complex<double>,1> giom((complex<double>*)info_N.ptr, info_N.shape[0], bl::neverDeleteData);

	svd.MatsubaraFrequency(giom, g, iOm, statistics);
	return _giom_;
      })
    .def("MatsubaraFrequencyDebug", [](class SVDFunc& svd, py::array_t<double>& _g_, py::array_t<int>& _iOm_, const string& statistics="fermi")->py::array_t<complex<double>>{
	//void MatsubaraFrequency(bl::Array<complex<double>,1>& giom, const bl::Array<double,1>& g, const bl::Array<int,1> iw, const string& statistics="fermi"){
	py::buffer_info info_w = _iOm_.request();
	bl::Array<int,1> iOm((int*)info_w.ptr, info_w.shape[0], bl::neverDeleteData);
	py::buffer_info info_g = _g_.request();
	bl::Array<double,1> g((double*)info_g.ptr, info_g.shape[0], bl::neverDeleteData);
	
	py::array_t<complex<double>> _giom_( iOm.size() );
	py::buffer_info info_N = _giom_.request();
	bl::Array<complex<double>,1> giom((complex<double>*)info_N.ptr, info_N.shape[0], bl::neverDeleteData);

	svd.MatsubaraFrequencyDebug(giom, g, iOm, statistics);
	return _giom_;
      })
    ;

}
