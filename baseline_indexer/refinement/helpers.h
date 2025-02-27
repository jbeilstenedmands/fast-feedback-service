#ifndef DIALS_RESEARCH_HELPERS
#define DIALS_RESEARCH_HELPERS

#include <gemmi/math.hpp> // for symmetric 3x3 matrix SMat33

#include <Eigen/Dense>
#include <gemmi/symmetry.hpp>
#include <gemmi/cellred.hpp>
#include <gemmi/scaling.hpp> // for constraints

using Eigen::Matrix3d;
using Eigen::Vector3d;

// Define a crystal orientation class - based on cctbx/crystal_orientation.h
class crystal_orientation {
public:
  crystal_orientation(){};
  crystal_orientation(Matrix3d const& B) : Astar(B){} // reciprocal=True

  // Astar  = 
  Matrix3d reciprocal_matrix() const {return Astar;}
private:
  Matrix3d Astar;
};

// Define the AG converter
struct AG {// convert orientation matrix A to metrical matrix g & reverse
  /*The general orientation matrix A is re-expressed in terms of the
    upper-triangular fractionalization matrix F by means of the following
    transformation:
                           F = (D * C * B * A).transpose()
    where D,C,B are three rotation matrices.
  */
  crystal_orientation orientation;
  double phi,psi,theta; //in radians
  Matrix3d B,C,D,F;
  gemmi::SMat33<double> G;
  void forward(crystal_orientation const& ori){
    orientation = ori;
    Matrix3d A(ori.reciprocal_matrix()); // i.e. B matrix (unhelpfully called A here)
    phi = std::atan2(A(0,2),-A(2,2));
    B = Matrix3d({
      {std::cos(phi),0.,std::sin(phi)},
      {0.,1.,0.},
      {-std::sin(phi),0.,std::cos(phi)}
    });
    Matrix3d BA(B * A);
    psi = std::atan2(-BA(1,2),BA(2,2));
    C = Matrix3d({
      {1.,0.,0.},
      {0., std::cos(psi),std::sin(psi)},
      {0.,-std::sin(psi),std::cos(psi)}
    });
    Matrix3d CBA (C * BA);

    theta = std::atan2(-CBA(0,1),CBA(1,1));
    D = Matrix3d({
      {std::cos(theta),std::sin(theta),0.},
      {-std::sin(theta),std::cos(theta),0.},
      {0.,0.,1.}
    });
    F = (D * CBA).transpose();
    Matrix3d G9 (A.transpose()*A); //3x3 form of metrical matrix
    G = {G9(0,0),G9(1,1),G9(2,2),G9(0,1),G9(0,2),G9(1,2)};
  }
  void validate_and_setG(gemmi::SMat33<double> const& g){
    // skip validation
    G = {g.u11, g.u22, g.u33, g.u12, g.u13, g.u23};
  }
  Matrix3d back() const {
    gemmi::GruberVector gv({G.u11, G.u22, G.u33, 2*G.u12, 2*G.u13, 2*G.u23});
    gemmi::UnitCell cell(gv.cell_parameters());
    gemmi::Mat33 F = cell.frac.mat;
    Matrix3d Fback{{F.a[0][0], F.a[0][1], F.a[0][2]},
              {F.a[1][0], F.a[1][1], F.a[1][2]},
              {F.a[2][0], F.a[2][1], F.a[2][2]}};
    return (B.inverse() * C.inverse() * D.inverse() * Fback);
  }
  crystal_orientation back_as_orientation() const {
    return crystal_orientation(back());
  }
};

// We're just working in P1 for now, so skip constraints
class Constraints {
public:
  Constraints(){}
  Constraints(gemmi::SpaceGroup space_group) : space_group_(space_group){}//note working in reciprocal space
  std::array<double, 6> independent_params(gemmi::SMat33<double> G){
    std::array<double, 6> params{{G.u11, G.u22, G.u33, G.u12, G.u13, G.u23}};
    return params;
  }
  gemmi::SMat33<double> all_params(std::array<double, 6> independent){
    return gemmi::SMat33<double>{independent[0], independent[1], independent[2], independent[3], independent[4], independent[5]};
  }
  int n_independent_params(){return 6;}
  std::array<double, 6> independent_gradients(gemmi::SMat33<double> all_grads){
    std::array<double, 6> grads = {
        all_grads.u11, all_grads.u22, all_grads.u33, all_grads.u12, all_grads.u13,all_grads.u23};
    return grads;
  }
private:
  gemmi::SpaceGroup space_group_;
};

#endif  // DIALS_RESEARCH_HELPERS