#include "internal/se3_schur_solve.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

int main() {
    using Matrix = Eigen::Matrix<double,6,6>;
    using Vector = Eigen::Matrix<double,6,1>;
    std::mt19937_64 rng(2817);
    std::normal_distribution<double> normal;
    double worst = 0;
    for(int trial=0; trial<12000; ++trial) {
        Matrix random, cross;
        Vector rhs, units, values;
        for(int i=0; i<36; ++i) { random.data()[i]=normal(rng); cross.data()[i]=normal(rng); }
        for(int i=0; i<6; ++i) {
            rhs[i]=normal(rng);
            units[i]=std::pow(10., ((trial+3*i)%13-6)/2.);
            values[i]=std::pow(10., double(i)*(trial%7)/5.);
        }
        const Matrix orthogonal=random.householderQr().householderQ();
        const Matrix q=units.asDiagonal()*orthogonal*values.asDiagonal()*orthogonal.transpose()*units.asDiagonal();
        Eigen::LLT<Matrix> llt(q);
        if(llt.info()!=Eigen::Success) throw std::runtime_error("Test cavity not SPD");
        const Matrix lower=llt.matrixL();
        const Matrix reference=cross.transpose()*llt.solve(cross);
        const Vector eta_reference=cross.transpose()*llt.solve(rhs);
        const Matrix diagonal=reference+Matrix::Identity();
        double packed[21], diag[21];
        for(int col=0,k=0; col<6; ++col) for(int row=0; row<=col; ++row,++k) {
            packed[k]=lower(col,row); diag[k]=diagonal(row,col);
        }
        std::array<double,44> full, compact;
        full.fill(1234567.); compact.fill(1234567.);
        slam::forwardSpd6CrossEtaReciprocal(lower.data(),cross.data(),rhs.data(),full.data()+1);
        slam::forwardSpd6CrossEtaReciprocal<true>(packed,cross.data(),rhs.data(),compact.data()+1);
        if(full.front()!=1234567. || full.back()!=1234567. || compact.front()!=1234567. || compact.back()!=1234567.)
            throw std::runtime_error("Forward output overwrite");
        if(full!=compact) throw std::runtime_error("Packed/dense forward mismatch");
        double precision[21], eta[6];
        slam::projectForwardSchur6(diag,full.data()+1,precision,eta);
        const Matrix expected=diagonal-reference;
        for(int col=0,k=0; col<6; ++col) {
            const double error=std::abs(eta[col]-eta_reference[col])/std::max(1.,eta_reference.norm());
            if(!std::isfinite(error)) throw std::runtime_error("Nonfinite eta");
            worst=std::max(worst,error);
            for(int row=0; row<=col; ++row,++k) {
                const double error=std::abs(precision[k]-expected(row,col))/std::max(1.,diagonal.norm());
                if(!std::isfinite(error)) throw std::runtime_error("Nonfinite precision");
                worst=std::max(worst,error);
                long double gram=0,absolute_sum=0;
                for(int i=0;i<6;++i) {
                    const long double product=static_cast<long double>(full[1+6*row+i])*full[1+6*col+i];
                    gram+=product;absolute_sum+=std::abs(product);
                }
                const long double projected=static_cast<long double>(diag[k])-gram;
                const long double bound=64*std::numeric_limits<double>::epsilon()*
                    (1+std::abs(static_cast<long double>(diag[k]))+absolute_sum);
                if(std::abs(static_cast<long double>(precision[k])-projected)>bound)
                    throw std::runtime_error("Component-scaled Gram projection regression");
            }
        }
    }
    // Mixed column scales must not let one huge diagonal mask small entries.
    for(double scale:{1e-4,1.,1e4,1e8}) {
        Matrix lower=Matrix::Identity(),cross=Matrix::Identity();Vector rhs=Vector::Ones();
        lower(2,0)=.125;lower(4,1)=-.25;cross.col(0)*=scale;cross(1,2)=.125;
        double forward[42],diagonal[21],precision[21],eta[6];
        slam::forwardSpd6CrossEtaReciprocal(lower.data(),cross.data(),rhs.data(),forward);
        for(int col=0,k=0;col<6;++col)for(int row=0;row<=col;++row,++k) {
            long double gram=0;for(int i=0;i<6;++i)gram+=static_cast<long double>(forward[6*row+i])*forward[6*col+i];
            diagonal[k]=static_cast<double>(gram)+(row==col?1.:0.);
        }
        slam::projectForwardSchur6(diagonal,forward,precision,eta);
        for(int col=0,k=0;col<6;++col)for(int row=0;row<=col;++row,++k) {
            long double gram=0,terms=0;
            for(int i=0;i<6;++i) {const long double term=static_cast<long double>(forward[6*row+i])*forward[6*col+i];gram+=term;terms+=std::abs(term);}
            const long double expected=static_cast<long double>(diagonal[k])-gram;
            const long double bound=64*std::numeric_limits<double>::epsilon()*(1+std::abs(static_cast<long double>(diagonal[k]))+terms);
            if(std::abs(static_cast<long double>(precision[k])-expected)>bound)
                throw std::runtime_error("Cancellation-sensitive projection regression");
        }
    }
    std::cout << "12000 SPD systems; forward-Gram normalized error=" << worst << '\n';
    if(worst>1e-9) throw std::runtime_error("Forward Schur regression");
}
