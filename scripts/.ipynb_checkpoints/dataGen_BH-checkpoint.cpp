/*

    Bethe-Heitler C++ data generator 

    - TCS divided by Z^2
    - iCDF for g+/k in interval [0.5,1]
    - no OpenMP 

    C++ version by Óscar Amaro 23 Mar 2023
    original python code by B Martinez

g++ -std=c++11 -I /Users/user/opt/include/boost dataBH.cpp -o dataBH
./dataBH

*/

#include <iostream> // standard library
#include <fstream>
#include <cmath> // sqrt,pow,etc.
#include <math.h>
#include <boost/math/special_functions/bessel.hpp> // For Bessel functions
#include <boost/math/quadrature/exp_sinh.hpp> // For integrating until infinity
#include <boost/math/quadrature/tanh_sinh.hpp> // For integrating with singularities
#include <boost/math/quadrature/gauss_kronrod.hpp> // For integrating well behaved functions

#include "/usr/local/opt/libomp/include/omp.h"
#include<ctime>
#include <chrono>
#include <thread>


using namespace std;

namespace constants
{
    // physical constants
    double epsilon_0 = 8.8541878128e-12; // [F m^-1]
    double c = 299792458.0; // [m s^-1] speed of light in vacuum
    double e = 1.602176634e-19; // [C] elementary charge
    double k = 1.380649e-23; // [J K^-1] boltzman constant
    double alpha = 0.0072973525693; // []
    double N_A = 6.02214076e+23; // [mol^-1]
    double h = 6.62607015e-34; // [J Hz^-1]
    double hbar = 1.054571817e-34; // [J s]
    double mu_0 = M_PI*4e-7; // [?]
    double m_e = 9.1093837015e-31; // [kg]
    double r_e = 2.8179493227e-15; // [m]
}

double ltf(double Z)
{
    /*
    This function determines the Thomas Fermi length for a given atomic number Z
    inputs :
        Z is th atomic number
    outputs :
        result is the TF length, normalised by the compton wavelength
    */

    double compton_wavelength = constants::hbar / (constants::m_e * constants::c); 
    double length = (4. * M_PI * constants::epsilon_0 * constants::hbar*constants::hbar / (constants::m_e * constants::e*constants::e) * pow(Z,-1.0/3.0) );
    double result = 0.885 * length / compton_wavelength;

    return result;
}


double I1(double d, double l)
{

    /*
    This function computes the term I1 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I1
    */

    double q = 1.0;
    double T1 = l * d * ( atan(l * d) - atan(l) );
    double T2 = - (l*l / 2.) * pow((1. - d),2.0) / (1. + l*l);
    double T3 = (1. / 2.) * log((1. + l*l) / (1. + pow((l * d),2.0)));

    double result = q*q * (T1 + T2 +T3);

    return result;
}



double I2(double d, double l)
{
    /*
    This function computes the term I2 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I2
    */

    double q = 1.0;
    double T1 = 4. * pow((l * d),3.0) * (atan(d * l) - atan(l));
    double T2 = (1. + 3. * pow((l * d),2.0) ) * log((1. + l*l) / (1. + pow((l * d),2.0)));
    double T3 = (6. * pow(l*l*d,2.0)) * log(d) / (1. + l*l);
    double T4 = l*l * (d - 1.) * (d + 1. - 4. * l*l * d*d) / (1. + l*l);

    double result = 0.5 * q * (T1 + T2 + T3 + T4);
    
    return result;
}


double bh_cs_dif(double gp, double k, double Z)
{
    /*
    This function computes the differential Bethe-Heitler cross-section
    inputs :
        gp is the energy of the positron
        k is the energy of the photon
        Z is the atomic number
    outputs :
        result is the differential cross-section in m^2
    */

    double result = 0.0, q=0.0, ge=0.0, d=0.0, l=0.0, T1=0.0, T2=0.0, T3=0.0;

    if ((k >= 2.) && (gp >= 1.) && (gp <= k-1.)) {

        q  = 1.0;
        ge = k - gp;
        d  = k / (2.0 * gp * ge);
        l  = ltf(Z);

        T1 = 4. * Z*Z * constants::r_e*constants::r_e * constants::alpha / pow(k,3.0);
        T2 = (gp*gp + ge*ge) * (I1(d, l) + 1.0);
        T3 = (2. / 3.) * gp * ge * (I2(d, l) + 5. / 6.);
        
        result = T1 * (T2 + T3);
    }
    
    return result;
}


double bh_cs(double Z, double k)
{
    /*
    This function computes the total Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
    outputs :
        result is the total cross-section in m^2
    */

    double result = 0.0;

    if (k > 2.0){

        boost::math::quadrature::tanh_sinh<double> integrator;

        auto j = [k,Z](double gp) {return bh_cs_dif(gp, k, Z);};

        return integrator.integrate(j,1.0,k-1.0);

    }

    return result;
}


double bh_cdf(double Z, double k, double gp)
{

    /*
    This function computes the CDF of the Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
        gp is the energy of the positron
    outputs :
        result is the CDF of the Bethe-Heitler cs (will be normalized by bh_cs)
            no longer (no units and between 0 and 1 by definition)
    */

    double result = 0.0;

    if ((k >= 2.) && (gp >= 0.) && (gp <= 1.)) {
        gp =  1.0 + (k - 2.) * gp; // rescale gp to k
        //double denominator = bh_cs(Z, k);

        boost::math::quadrature::tanh_sinh<double> integrator;

        auto j = [k,Z](double gp) {return bh_cs_dif(gp, k, Z);};

        double numerator = integrator.integrate(j,1,gp);

        result = numerator ; /// denominator;
    }
    
    return result;
}

void update_progress_bar(double progress, std::chrono::steady_clock::time_point start_time) {
    // Define the width of the progress bar
    const int bar_width = 50;

    // Compute the number of completed bars and the remaining fraction
    const int num_completed_bars = static_cast<int>(progress * bar_width);
    const double remaining_fraction = progress * bar_width - num_completed_bars;

    // Compute the elapsed time
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;

    // Compute the estimated time until completion (ETA)
    const auto eta = elapsed_time / progress * (1 - progress);

    // Output the progress bar and ETA to the console
    std::cout << "[";

    for (int i = 0; i < num_completed_bars; ++i) {
        std::cout << "=";
    }

    if (num_completed_bars < bar_width) {
        std::cout << ">";

        for (int i = num_completed_bars + 1; i < bar_width; ++i) {
            std::cout << " ";
        }
    }

    std::cout << "] " << static_cast<int>(progress * 100.0) << "%";

    if (progress > 0) {
        std::cout << " (ETA: " << std::chrono::duration_cast<std::chrono::seconds>(eta).count() << "s)";
    }

    // Flush the output buffer to update the progress bar immediately
    std::cout << std::flush;
}


///////// iCDF
int main( int argc, char* argv[] )
{

    const int idim = 216; //216

    // parameters
    double Z = 0.0, k = 0.0;
    double Zmin = 1, Zmax = 50;
    double kmin = 2.1e0, kmax = 2.0e4;

    double gp=0.0, cdf=0.0, norm=0.0;
    double gpmin = 0.5, gpmax = 1-1e-4;
    
    // compare with benchmarkBH.ipynb

    std::ofstream myfile2;
    myfile2.open ("BH_cpp_icdf[051]_1e7_unbal_unprep.csv");

    std::cout << "start icdf:";
    int time_before_loop_begins = time(NULL);
    //progressbar bar2(idim);

    // Get the current time
    const auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < idim; i++){

        const double progress = static_cast<double>(i) / static_cast<double>(idim);
        update_progress_bar(progress, start_time);

        for (int j = 0; j < idim; j++){

            // calculate total cross section for (Z,k)
            Z = i/double(idim) *(Zmax-Zmin) + Zmin; // linspace
            k = pow(10.0, j/double(idim) *(log10(kmax)-log10(kmin)) + log10(kmin)); // logspace
            norm = bh_cs(Z, k);

            for (int kk = 0; kk < idim; kk++){
                gp = kk/double(idim) *(gpmax-gpmin) + gpmin; // linspace
                cdf = bh_cdf(Z, k, gp)/norm;
                myfile2 << Z << "," << k << "," << cdf << "," << gp << "\n";
            }
        }

        //bar2.update();
        // Move the cursor to the beginning of the line to overwrite the current progress bar
        std::cout << "\r";
    }
    myfile2.close();
    std::cout << std::endl;

    std::cout << "done";
    int time_after_loop_ends = time(NULL);
    int time_diff = time_after_loop_ends - time_before_loop_begins;
    cout << "Time taken to run loop = " << time_diff << " seconds.";

    return 0;
}

///////// TCS
/* compare with benchmarkBH.ipynb
std::cout << ltf(29) << "\n";
std::cout << I1(0.4, 0.2) << "\n";
std::cout << I2(0.4, 0.2) << "\n";
std::cout << bh_cs_dif(500, 1000, 29) << "\n";
std::cout << bh_cs(29, 1000) << "\n";
std::cout << bh_cdf(29, 1000, 0.99)  << "\n";

39.47407317354477
9.223024053281806e-05
8.033356412162976e-05
5.246356762135083e-31
5.828960814393803e-28
0.9932705824571192
*/


///////// TCS
/*
int main( int argc, char* argv[] )
{

    // total cross section
    int idim = 3000.0;

    double Z = 0.0, k = 0.0;
    double Zmin = 1, Zmax = 100;
    double kmin = 2.1e0, kmax = 2.0e4;

    std::ofstream myfile;
    myfile.open ("BHcpp_tcs.csv");

    std::cout << "start tcs:";
    int time_before_loop_begins = time(NULL);
    progressbar bar(idim);

        for (int i = 0; i < idim; i++){
            for (int j = 0; j < idim; j++){
                Z = i/double(idim) *(Zmax-Zmin) + Zmin; // linspace
                k = pow(10.0, j/double(idim) *(log10(kmax)-log10(kmin)) + log10(kmin)); // logspace
                myfile << Z << "," << k << "," << bh_cs(Z, k) << "\n";
            }
            bar.update();
        }
        std::cout << "done";
    int time_after_loop_ends = time(NULL);
    int time_diff = time_after_loop_ends - time_before_loop_begins;
    cout << "Time taken to run loop = " << time_diff << " seconds.";

    myfile.close();

    return 0;
}
*/
