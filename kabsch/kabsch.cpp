#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <math.h>
#include <limits>
#include <chrono>

#define FNAME_ACCURACY "accuracy.txt"
#define FNAME_RUNTIME "runtime.txt"

using std::cos;

// temporary declarations
namespace jp
{
	typedef std::pair<cv::Mat, cv::Mat> cv_trans_t;
}

inline bool containsNaNs(const cv::Mat& m)
{
	return cv::sum(cv::Mat(m != m))[0] > 0;
}

void print(std::string name, cv::Mat& mat)
{
	std::cout << "matrix: " << name << std::endl;
	std::cout << "rows: " << mat.rows << std::endl;
	std::cout << "cols: " << mat.cols << std::endl;
	std::cout << "channels: " << mat.channels() << std::endl;
	std::cout << mat << std::endl << std::endl;
}
////////

/*
 * @brief Transformation class that stores rotation and translation vectors
 *
 */
class Transformation
{
public:

	Transformation(cv::Mat& rots, cv::Mat& trans)
	{
		// Args:
		// 		rots: user-defined rotational angles,
		// 				[angleX, angleY, angleZ], 0 ,= angel < 2*PI
		// 		trans: user-defined translation
		this->rots = cv::Mat_<float>::zeros(1, 3);
		r = cv::Mat_<float>::zeros(1, 3);
		t = cv::Mat_<float>::zeros(1, 3);

		if (rots.empty())
			cv::randu(this->rots, cv::Scalar(0), cv::Scalar(2*M_PI));
		else
			rots.copyTo(this->rots);

		if (trans.empty())
			cv::randn(t, 0, 1);
		else
			trans.copyTo(t);

		cv::Rodrigues(computeRotationMatrix(), r);

	}

	cv::Mat computeRotationMatrix()
	{
		// Returns a rotation matrix based on rotational angles
		cv::Mat rotX = (cv::Mat_<float> (3, 3) <<
						1, 0, 0,
						0, cos(rots.at<float>(0,0)), -sin(rots.at<float>(0,0)),
						0, sin(rots.at<float>(0,0)), cos(rots.at<float>(0,0))
						);
		cv::Mat rotY = (cv::Mat_<float> (3, 3) <<
						cos(rots.at<float>(0,1)), 0, -sin(rots.at<float>(0,1)),
						0, 1, 0,
						sin(rots.at<float>(0,1)), 0, cos(rots.at<float>(0,1))
						);
		cv::Mat rotZ = (cv::Mat_<float> (3, 3) <<
						cos(rots.at<float>(0,2)), -sin(rots.at<float>(0,2)), 0,
						sin(rots.at<float>(0,2)), cos(rots.at<float>(0,2)), 0,
						0, 0, 1
						);

		return rotZ * (rotY * rotX);
	}

	cv::Mat getRotationAngles()
	{
		// Returns angles w.r.t to current rotation matrix
		return rots;
	}

	cv::Mat getRotationVector()
	{
		// Returns angle-axis rotation vector
		return r;
	}

	cv::Mat getRotationMatrix()
	{
		// Returns rotation matrix
		cv::Mat R;
		cv::Rodrigues(r, R);
		return R;
	}

	cv::Mat getTranslation()
	{
		// Returns translation vector
		return t;
	}

	void setRotationAngles(cv::Mat& rots)
	{
		// Sets the rotation vector
		// Args:
		// 		rots - angles of rotations
		this->rots = rots.clone();
		cv::Rodrigues(computeRotationMatrix(), r);
	}

	void setTranslation(cv::Mat& trans)
	{
		// Sets the translation vector
		t = trans.clone();
	}

private:
	cv::Mat rots;
	cv::Mat r;
	cv::Mat t;
};

/*
 * Helper functions
 */
/*
 * @brief Creates 3D scene point dataset
 *
 * @param N: number of points
 * @param seed: for random value generation
 * 			(<0: fixed value, =0: random seed, >0: fixed seed)
 * @return data: Nx3 matrix for scene point
 *
 */
std::vector<cv::Point3f> createScene(unsigned int N, int seed=-1)
{
//	std::cout << "create scene points" << std::endl;

	std::vector<cv::Point3f> data;

	if (seed < 0)
	{
//		if (N != 4)
//			std::cout << "Warning: returns fixed dataset of size 4." << std::endl;

//		data.push_back(cv::Point3f(.15, .2, .3));
//		data.push_back(cv::Point3f(.4, .5, .6));
//		data.push_back(cv::Point3f(.7, .8, .9));
//		data.push_back(cv::Point3f(.2, .1, .2));

		data.push_back(cv::Point3f(1., 0., 0.));
		data.push_back(cv::Point3f(0., 0., 0.));
		data.push_back(cv::Point3f(0., 1., 0.));
		data.push_back(cv::Point3f(1., 1., 0.));


		return data;
	}
	else
	{
		cv::RNG rng(seed);

		for (unsigned int i = 0; i < N; ++i)
		{
			data.push_back(cv::Point3f(rng.gaussian(1), rng.gaussian(1), rng.gaussian(1)));
		}
	}

	return data;
}

/*
 * @brief Given scene points, create measurements by some transformations
 *
 * @param data: scene points
 * @param transform: object which contains rotation and translation
 * @fixed:
 * 		- if true, existing transformation will be overriden by random transformation
 * 		- if false, use existing transformation
 *
 * @return tfData: Nx3 transformed scene points used as measurement
 *
 */
std::vector<cv::Point3f> createMeasurements(std::vector<cv::Point3f>& data, Transformation& transform, bool fixed=false)
{
//	std::cout << "create measurements " << std::endl;

	std::vector<cv::Point3f> measurements;

	// if not fixed, set a new transformation with random values
	if (!fixed)
	{
		// set rotational angles between 0 and 2*PI
		cv::Mat rots = cv::Mat::zeros(1, 3, CV_32F);
		cv::randu(rots, cv::Scalar(0), cv::Scalar(2*M_PI));
		transform.setRotationAngles(rots);

		// translation vector drawn from normal distribution
		cv::Mat trans = cv::Mat::zeros(1, 3, CV_32F);
		cv::randn(trans, 0, 1);
		transform.setTranslation(trans);
	}

	// P = RX + T
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		cv::Mat_<double> res= transform.getRotationMatrix() * cv::Mat(data[i], false);
		cv::add(res, transform.getTranslation().t(), res);
		measurements.push_back(cv::Point3f(res.at<double>(0,0), res.at<double>(1,0), res.at<double>(2,0)));
	}

	return measurements;
}

/*
 * @brief Computes mean squared differences of two matrices
 */
double compareMatrix(cv::Mat A, cv::Mat B)
{
	cv::Mat_<double> diff, diff_sq;
	cv::subtract(A, B, diff);
	cv::pow(diff, 2, diff_sq);
	return cv::sum(diff_sq)[0] / diff_sq.total();
}

/*
 * @brief Writes the contents into file in specified format
 *
 * @param filename: name of output file
 * @param mode: mode to open the file
 * @param format: format of contents, "header", "table", "default"
 * @param contents: contents of the file
 *
 */
void writeToFile(std::string filename, std::ios_base::openmode mode, std::string format, std::vector<std::pair<std::string, double>> contents)
{
	std::ofstream fh(filename, mode);

	if (fh.is_open())
	{
		if (format == "table")
		{
			for (auto iter = contents.begin(); iter != contents.end(); ++iter)
			{
				fh << (*iter).second << "\t";
			}
			fh << std::endl;
		}
		else if (format == "header")
		{
			for (auto iter = contents.begin(); iter != contents.end(); ++iter)
			{
				fh << (*iter).first << "\t";
			}
			fh << std::endl;
		}
		else if (format == "default")
		{
			for (auto iter = contents.begin(); iter != contents.end(); ++iter)
			{
				fh << (*iter).first << std::endl;
				fh << (*iter).second << std::endl << std::endl;
			}
		}

		fh.close();
	}
	else
		std::cout <<"Unable to open file " << filename << std::endl;
}


cv::Mat svd_backward(const std::vector<cv::Mat> &grads, const cv::Mat& self, bool some, const cv::Mat& raw_u, const cv::Mat& sigma, const cv::Mat& raw_v)
{
	auto m = self.rows;
	auto n = self.cols;
	auto k = sigma.cols;
	auto gsigma = grads[1];

	auto u = raw_u;
	auto v = raw_v;
	auto gu = grads[0];
	auto gv = grads[2];
/*
	if (!some)
	{
		// We ignore the free subspace here because possible base vectors cancel
		// each other, e.g., both -v and +v are valid base for a dimension.
		// Don't assume behavior of any particular implementation of svd.
	    u = raw_u.narrow(1, 0, k);
	    v = raw_v.narrow(1, 0, k);
	    if (gu.defined())
	    {
	    	gu = gu.narrow(1, 0, k);
	    }
	    if (gv.defined())
	    {
	    	gv = gv.narrow(1, 0, k);
	    }
	}
*/
	auto vt = v.t();

	cv::Mat sigma_term;

	if (!gsigma.empty())
	{
		sigma_term = (u * cv::Mat::diag(gsigma)) * vt;
	}
	else
	{
		sigma_term = cv::Mat::zeros(self.size(), self.type());
	}
	// in case that there are no gu and gv, we can avoid the series of kernel
	// calls below
	if (gv.empty() && gu.empty())
	{
		return sigma_term;
	}

	auto ut = u.t();
	auto im = cv::Mat::eye((int)m, (int)m, self.type());
	auto in = cv::Mat::eye((int)n, (int)n, self.type());
	auto sigma_mat = cv::Mat::diag(sigma);

	cv::Mat sigma_mat_inv;
	cv::pow(sigma, -1, sigma_mat_inv);
	sigma_mat_inv = cv::Mat::diag(sigma_mat_inv);

	cv::Mat sigma_sq, sigma_expanded_sq;
	cv::pow(sigma, 2, sigma_sq);
	sigma_expanded_sq = cv::repeat(sigma_sq, sigma_mat.rows, 1);

	cv::Mat F = sigma_expanded_sq - sigma_expanded_sq.t();
	// The following two lines invert values of F, and fills the diagonal with 0s.
	// Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
	// first to prevent nan from appearing in backward of this function.
	F.diag().setTo(std::numeric_limits<float>::max());
	cv::pow(F, -1, F);

	cv::Mat u_term, v_term;

	if (!gu.empty())
	{
		cv::multiply(F, ut*gu-gu.t()*u, u_term);
		u_term = (u * u_term) * sigma_mat;
		if (m > k)
		{
			u_term = u_term + ((im - u*ut)*gu)*sigma_mat_inv;
		}
		u_term = u_term * vt;
	}
	else
	{
		u_term = cv::Mat::zeros(self.size(), self.type());
	}

	if (!gv.empty())
	{
		auto gvt = gv.t();
		cv::multiply(F, vt*gv - gvt*v, v_term);
		v_term = (sigma_mat*v_term) * vt;
		if (n > k)
		{
			v_term = v_term + sigma_mat_inv*(gvt*(in - v*vt));
		}
		v_term = u * v_term;
	}
	else
	{
		v_term = cv::Mat::zeros(self.size(), self.type());
	}

	return u_term + sigma_term + v_term;
}

/*
 * @brief
 *
 * @param imgdPts: measurements
 * @param objPts: scene points
 * @param extCam Output parameter: extrinsic camera matrix (incl. rotation and translation)
 * @param jacobean Output parameter: 6x3N jacobean matrix of rotation and translation vector
 * 					w.r.t scene point coordinates
 * @param calc: flag to indicate calcuation of jacobean matrix
 *
 */
bool kabsch(std::vector<cv::Point3f>& imgdPts, std::vector<cv::Point3f>& objPts, jp::cv_trans_t& extCam, cv::Mat& jacobean, bool calc=false)
{

	cv::Mat P, X, A, U, W, Vt, D, R, V;
	cv::Mat ctrCoeff, N_inv_1xN, N_inv_NxN, gRodr;
	cv::Mat& r = extCam.first;
	cv::Mat& t = extCam.second;

	unsigned int N = objPts.size();

	P = cv::Mat(imgdPts, false).reshape(1, N);

	X = cv::Mat(objPts, false).reshape(1, N);

	N_inv_1xN = cv::Mat(1, N, CV_32F, 1.f/N);

	cv::Mat tx = N_inv_1xN * X;
	cv::Mat tp = N_inv_1xN * P;

	//ctrCoeff = cv::Mat::eye(N,N, CV_32F) - cv::Mat(N, N, CV_32F, 1.f/N);

	cv::Mat Xc =  X - cv::repeat(tx, N, 1); //ctrCoeff * X;
	cv::Mat Pc =  P - cv::repeat(tp, N, 1); //ctrCoeff * P;

	A = Pc.t() * Xc;

	cv::SVD::compute(A, W, U, Vt);

	bool degenerate = false;

	if ((unsigned int)cv::countNonZero(W) != (unsigned int)W.total())
		degenerate = true;

	if (std::abs(W.at<float>(0,0)-W.at<float>(1,0)) < 1e-6
			|| std::abs(W.at<float>(0,0)-W.at<float>(2,0)) < 1e-6
			|| std::abs(W.at<float>(1,0)-W.at<float>(2,0)) < 1e-6)
		degenerate = true;

	if (calc && degenerate)
		return true;

	float d = cv::determinant(U * Vt);

	D = (cv::Mat_<float>(3,3) <<
				1., 0., 0.,
				0., 1., 0.,
				0., 0., d );

	R = U * (D * Vt);

	if (calc)
	{
		cv::Rodrigues(R, r, gRodr);
	}
	else
	{
		cv::Rodrigues(R, r);
	}


	t = tp - (R*tx.t()).t();

	r = r.reshape(1, 3);
	t = t.reshape(1, 3);
//	print("r", r);
//	print("t", t);

	if (!calc)
		return true;

	cv::Mat onehot;
	cv::Mat drdR, dRdU, dRdVt, dRidU, dRidVt, dRidV, dRidA, dAdPct, dAdXc, dRidXc, dRidX, drdX, dtdR, dtdtx, dtdX;
	cv::Mat dRdX, dtxdX, dtxdninv;//, dXcdcoeff, dXcdX;

	dRdX = cv::Mat_<float>::zeros(9, N * 3);
	jacobean = cv::Mat_<double>::zeros(6, N * 3);

	cv::matMulDeriv(U, D*Vt, dRdU, dRdVt);
	cv::matMulDeriv(Pc.t(), Xc, dAdPct, dAdXc);
	cv::matMulDeriv(R, tx.t(), dtdR, dtdtx);
	cv::matMulDeriv(N_inv_1xN, X, dtxdninv, dtxdX);
//	std::cout << ctrCoeff.total() << " "<< X.total() << std::endl;
//	print("ctrCoeff", ctrCoeff);
//	print("X", X);
//	cv::matMulDeriv(ctrCoeff, X, dXcdcoeff, dXcdX);


//	print("dtdR", dtdR);
//	print("dtdtx", dtxdX);

	dRdU = dRdU.t();
	dRdVt = dRdVt.t();
//	dAdPct = dAdPct.t();
	dAdXc = dAdXc.t(); //9x3N->3Nx9
//	dXcdX = dXcdX.t();

	V = Vt.t();
	W = W.reshape(1, 1);

	for (int i = 0; i < 9; ++i)
	{
		onehot = cv::Mat::zeros(9, 1, CV_32F);
		onehot.at<float>(i, 0) = 1;

		dRidU = dRdU * onehot;
		dRidVt = dRdVt * onehot;

		dRidU = dRidU.reshape(1, 3);
		dRidVt = dRidVt.reshape(1, 3);

		dRidVt = D*dRidVt;

		dRidV = dRidVt.t();

		std::vector<cv::Mat> grads{dRidU, cv::Mat(), dRidV};

		dRidA = svd_backward(grads, A, true, U, W, V);


		dRidA = dRidA.reshape(1, 9);

		dRidXc = dAdXc * dRidA; // 3Nx1
		dRidXc = dRidXc.reshape(1, N);

		dRidX = cv::Mat::zeros(dRidXc.size(), dRidXc.type());

		for (int m = 0; m < dRidXc.rows; ++m)
		{
			for (int n = 0; n < dRidXc.cols; ++n)
			{
				for (int l = 0; l < dRidXc.rows; ++l)
				{

					if (l == m)
					{
						dRidX.at<float>(m,n) += dRidXc.at<float>(l,n) * (N-1) / N;
					}
					else
					{
						dRidX.at<float>(m,n) -= dRidXc.at<float>(l,n) / N;
					}

				}

			}
		}

//		dRidX = dXcdX * dRidXc;

		dRidX = dRidX.reshape(1, 1);

		dRidX.copyTo(dRdX.rowRange(i, i+1));
	}

	drdX = gRodr.t() * dRdX;

	drdX.copyTo(jacobean.rowRange(0, 3));

	dtdX = - dtdR * dRdX - dtdtx * dtxdX;

	dtdX.copyTo(jacobean.rowRange(3, 6));

//	print("jacobian", jac);

	return true;

}

/*
 * @brief Performs Kabsch algorithm and differentiation
 * 			using central finite differences
 *
 * @param imgdPts: measurement points
 * @param objPts: scene points
 * @param extCam Output parameter: extrinsic camera matrix (incl. rotation and translation)
 * @param eps: step size in finite difference approximation
 *
 * @return 6xN jacobean matrix w.r.t scene point coordinates
 *
 */
cv::Mat_<double> kabsch_fd(std::vector<cv::Point3f>& imgdPts, std::vector<cv::Point3f> objPts, jp::cv_trans_t& extCam, float eps = 0.001f)
{
	cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, objPts.size()*3);
	bool success;
	cv::Mat jac;

	kabsch(imgdPts, objPts, extCam, jac);

	for (unsigned int i = 0; i < objPts.size(); ++i)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// forward step

			jp::cv_trans_t fStep;
			success = kabsch(imgdPts, objPts, fStep, jac); //return success flag?

			if(!success)
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

			if(j == 0) objPts[i].x -= 2 * eps;
			else if(j == 1) objPts[i].y -= 2 * eps;
			else if(j == 2) objPts[i].z -= 2 * eps;

			// backward step
			jp::cv_trans_t bStep;
			success = kabsch(imgdPts, objPts, bStep, jac); //return success flag?

			if(!success)
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// gradient calculation
			fStep.first = (fStep.first - bStep.first) / (2 * eps);
			fStep.second = (fStep.second - bStep.second) / (2 * eps);

			fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
			fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

			if(containsNaNs(jacobean.col(i * 3 + j)))
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

		}
	}

	return jacobean;

}


/*
 * Test functions
 */
/*
 * @brief
 */
void test_runtime()
{
	std::cout << "Runtime test." << std::endl;
	writeToFile(FNAME_RUNTIME, std::ofstream::out, "header",
	    		{{"#Points", 0}, {"#Trials", 0}, {"FiniteDifference(sec)", 0},
	    		{"AnalyticalGradient(sec)", 0}});

	std::vector<unsigned int> powers {2, 4, 6, 8, 10};
	std::vector<unsigned int> N; //number of points by 2^power
	unsigned int trials = 50; //number of repetitions per dataset size

	cv::Mat angles, trans;
	Transformation tf(angles, trans);
//	std::cout << "Transformation set" << std::endl;

	// sum of time for all the trials of each dataset size
	double sumFdTime = 0, sumAgTime = 0;

	// averaged runtime for each dataset size
	std::vector<double> timeFd, timeAg;

	std::vector<cv::Point3f> scenePts, measurePts;
	jp::cv_trans_t extCamAg, extCamFd;
	cv::Mat jacAg;

	for (unsigned int i = 0; i < powers.size(); ++i)
	{
		// compute dataset size
		N.push_back(std::pow(2, powers[i]));
		std::cout << N[i] << std::endl;

		// create sample scene points
		scenePts = createScene(N[i], 10);
//		std::cout << "Scene points = " << std::endl << " " << scenePts << std::endl;

		//create measurements
		measurePts = createMeasurements(scenePts, tf, false);
//		std::cout << "Mesurements = " << std::endl << " " << measurePts << std::endl;


		for (unsigned int j = 0; j < trials; ++j)
		{


			auto beginAg = std::chrono::high_resolution_clock::now();
			kabsch(measurePts, scenePts, extCamAg, jacAg, true);
			auto endAg = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> secAg = endAg-beginAg;
			sumAgTime += secAg.count();

			auto beginFd = std::chrono::high_resolution_clock::now();
			kabsch_fd(measurePts, scenePts, extCamFd);
			auto endFd = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> secFd = endFd-beginFd;
			sumFdTime += secFd.count();
		}

		timeAg.push_back(sumAgTime/trials);
		timeFd.push_back(sumFdTime/trials);

		sumAgTime = 0;
		sumFdTime = 0;

		writeToFile(FNAME_RUNTIME, std::ofstream::app, "table",
        		{{"#Points", N[i]}, {"#Trials", trials}, {"FiniteDifference(sec)", timeFd[i]},
        		{"AnalyticalGradient(sec)", timeAg[i]}});
	}

	std::cout << "Finished run time tests." << std::endl;

}

/*
 * @brief
 */
void test_accuracy()
{
	std::cout << "Accuracy test." << std::endl;

	std::vector<unsigned int> powers {2, 4, 6, 8, 10};
	std::vector<unsigned int> N; //number of points by 2^power
	unsigned int trials = 50; //trials per dataset size
	float etol = 1e-3; //error tolerance of results
	float delta = 0.001f; //step size used in finite difference calculation

	// number of cases in all trials per dataset size
	unsigned int cntErrJac = 0, cntErrRot = 0, cntErrTrans = 0, cntErrRotFD = 0, cntDegen = 0;
	// sum of errors in all trials per dataset size
	unsigned int sumErrJac = 0, sumErrRot = 0, sumErrTrans = 0, sumErrRotFD = 0;

	// create a known rotation and translation
	cv::Mat angles = (cv::Mat_<float>(1,3) << 0., M_PI/2., 0.);//(cv::Mat_<float>(1,3) << 0., M_PI/2., M_PI/3.);
	cv::Mat trans = (cv::Mat_<float>(1,3) << 0., 0., 0.);
	Transformation tf(angles, trans);

	std::vector<cv::Point3f> scenePts, measurePts;


	cv::Mat jacAg, jacFd;

	writeToFile(FNAME_ACCURACY, std::ofstream::out, "header",
	    		{{"#Points", 0}, {"#Trials", 0}, {"#Degeneracy", 0},
	    		{"#Jacobian errors", 0}, {"#Tranlation errors", 0},
	    		{"#Rotation errors", 0}, {"#Rotation errors (Finite Difference)", 0},
	    		{"Jacobian error", 0}, {"Translation error", 0},
	    		{"Rotation erorr", 0}});

	for (unsigned int i = 0; i < powers.size(); ++i)
	{
		// compute dataset size
		N.push_back(std::pow(2, powers[i]));
		std::cout << N[i] << std::endl;

		// scene points created here will be fixed in all the trials later
//		scenePts = createScene(N[i], 10);

		for (unsigned int j = 0; j < trials; ++j)
		{
			jp::cv_trans_t extCamAg, extCamFd;

			// scene point created here will change in every trial
			scenePts = createScene(N[i], 10);
//			std::cout << "Scene points = " << std::endl << " " << scenePts << std::endl;

			// create measurements with random rotation and translation
			measurePts = createMeasurements(scenePts, tf, false);
//			std::cout << "Mesurements = " << std::endl << " " << measurePts << std::endl;

//			std::cout << "Translation = " << std::endl << " " << tf.getTranslation() << std::endl;
//			std::cout << "Rotation vector = " << std::endl << " " << tf.getRotationVector() << std::endl;
//			std::cout << "Rotation angles = " << std::endl << " " << tf.getRotationAngles() << std::endl;

			// run Kabsch with analytical gradients
			kabsch(measurePts, scenePts, extCamAg, jacAg, true);

			if (extCamAg.first.empty() && extCamAg.second.empty())
			{
				cntDegen += 1;
				jacAg = kabsch_fd(measurePts, scenePts, extCamAg, delta);
			}

			jacFd = kabsch_fd(measurePts, scenePts, extCamFd, delta);


//			std::cout << "Estimated rotation vector = " << std::endl << " " << extCamAg.first << std::endl;
//			std::cout << "Estimated translation vector = " << std::endl << " " << extCamAg.second << std::endl;

			double errJac = compareMatrix(jacAg, jacFd);
			double errRot = compareMatrix(tf.getRotationVector(), extCamAg.first);
			double errTrans = compareMatrix(tf.getTranslation(), extCamAg.second.t());
			double errRotFD = compareMatrix(tf.getRotationVector(), extCamFd.first);


			if (errJac > etol)
				cntErrJac += 1;

			if (errRot > etol)
				cntErrRot += 1;

			if (errTrans > etol)
				cntErrTrans += 1;

			if (errRotFD > etol)
				cntErrRotFD += 1;

			sumErrJac += errJac;
			sumErrRot += errRot;
			sumErrTrans += errTrans;
			sumErrRotFD += errRotFD;

		}

        sumErrJac /= trials;
        sumErrRot /= trials;
        sumErrTrans /= trials;

        writeToFile(FNAME_ACCURACY, std::ofstream::app, "table",
        		{{"#Points", N[i]}, {"#Trials", trials}, {"#Degeneracy", cntDegen},
        		{"#Jacobian errors", cntErrJac}, {"#Tranlation errors", cntErrTrans},
        		{"#Rotation errors", cntErrRot}, {"#Rotation errors (Finite Difference)", cntErrRotFD},
        		{"Jacobian error", sumErrJac}, {"Translation error", sumErrTrans},
        		{"Rotation erorr", sumErrRot}});

        sumErrJac = 0;
        sumErrRot = 0;
        sumErrTrans = 0;
        sumErrRotFD = 0;

        cntDegen = 0;
        cntErrJac = 0;
        cntErrRot = 0;
        cntErrTrans = 0;
        cntErrRotFD = 0;

	}

	std::cout << "Finish accuracy test." << std::endl;

}

void test_svd_backward()
{
	cv::Mat gU = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					0., 0., 0.7854,
					0.5554, -0.5554, 0.);
	cv::Mat gS;

	cv::Mat gV = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					0.5554, 0.5554, 0.,
					-0.5554, 0.5554, 0.);

	cv::Mat A = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					1., 2., 0.,
					2., 1., 0.);

	cv::Mat U = (cv::Mat_<float>(3,3) <<
					0., 0., 1.,
					0.7071, -0.7071, 0.,
					0.7071, 0.7071, 0. );

	cv::Mat S = (cv::Mat_<float>(1,3) <<
					3., 1., 0.);

	cv::Mat V = (cv::Mat_<float>(3,3) <<
					0.7071, 0.7071, 0.,
					0.7071, -0.7071, 0.,
					0., 0., 1. );

	std::vector<cv::Mat> grads{gU, gS, gV};

	cv::Mat res = svd_backward(grads, A, true, U, S, V);

	std::cout << "result: " << std::endl << " " << res << std::endl;
}

void test_mm_backward()
{
	cv::Mat P = (cv::Mat_<float>(3,4) <<
				2., 2., 2., 2.,
				5., 5., 5., 5.,
				7., 7., 7., 7.);

	cv::Mat X = (cv::Mat_<float>(4,3) <<
					12., 12., 12.,
					15., 15., 15.,
					17., 17., 17.,
					13., 13., 13.);

	cv::Mat dA, dB;

	cv::matMulDeriv(P, X, dA, dB);

	print("dA", dA);
	print("dB", dB);
}

int main(int argc, char** argv)
{

	std::cout << "Kabsch Algorithm..." << std::endl;

	test_runtime();

	test_accuracy();

//	test_svd_backward();

//	test_mm_backward();

//	float data[9] = {-0.2160, -0.6887,  0.0000, -0.7114,  0.5965,  0.0000, -0.6687, -0.4121,  0.0000};
//	cv::Mat M(3,3, CV_32F, data);
//	std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;
//
//	// OpenCV implementation
//	cv::Mat u, w, vt;
//	std::cout << "[start] OpenCV SVD decomposition" << std::endl;
//	cv::SVD::compute(M, w, u, vt);
//	std::cout << "[end] OpenCV SVD decomposition" << std::endl;
//	std::cout << "u = " << std::endl << " " << u << std::endl << std::endl;
//	std::cout << "w = " << std::endl << " " << w << std::endl << std::endl;
//	std::cout << "vt = " << std::endl << " " << vt << std::endl << std::endl;

	return 0;
}



//	std::cout << "test translation:" << std::endl << cv::repeat(cv::Mat(tf.getTranslation()).reshape(1, 1), 3, 1) << std::endl;
//
//	std::cout << "Rotation matrix = " << std::endl << " " << tf.getRotationMatrix() << std::endl;
//	std::cout << "Translation = " << std::endl << " " << tf.getTranslation() << std::endl;
//	std::cout << "Rotation vector = " << std::endl << " " << tf.getRotationVector() << std::endl;
//	std::cout << "Rotation angles = " << std::endl << " " << tf.getRotationAngles() << std::endl;
//	std::vector<cv::Point3f> scenePts = createScene(4);
//
//	std::cout << "Data = " << std::endl << " " << scenePts << std::endl;
//
//	std::vector<cv::Point3f> measurePts = createMeasurements(scenePts, tf);
//
//	std::cout << "Mesurements = " << std::endl << " " << measurePts << std::endl;
//
//	cv::Mat r_est, t_est, jacobian;
//
//	kabsch(measurePts, scenePts, r_est, t_est, jacobian, true);
//
//	std::cout << "Estimated rotation vector = " << std::endl << " " << r_est << std::endl;
//	std::cout << "Estimated translation vector = " << std::endl << " " << t_est << std::endl;

//	cv::Mat jacFd = kabsch_fd(measurePts, scenePts);

//	std::cout << "Jacobian (finite difference): " << std::endl << " " << jacFd << std::endl;



//		dRidX = cv::Mat::zeros(dRidXc.size(), dRidXc.type());
//
//		for (int m = 0; m < dRidXc.rows; ++m)
//		{
//			for (int n = 0; n < dRidXc.cols; ++n)
//			{
//				for (int l = 0; l < dRidXc.rows; ++l)
//				{
//
//					if (l == m)
//					{
//						dRidX.at<float>(m,n) += dRidXc.at<float>(l,n) * (N-1) / N;
//					}
//					else
//					{
//						dRidX.at<float>(m,n) -= dRidXc.at<float>(l,n) / N;
//					}
//
//				}
//
//			}
//		}

