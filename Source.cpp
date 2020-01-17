#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

struct sparsematrix
{
	vector<double> AV;
	vector<unsigned> ANR;
	vector<unsigned> ANC;
	unsigned n, nnz;
	void ReadFromFile(string path)
	{
		ifstream data;
		data.open(path);
		char s[255];
		do
		{
			data.getline(s, 255);
		} while ((s[0] == '%') && (!data.eof()));

		unsigned size[3];
		unsigned k = 0;
		for (unsigned i = 0; i < 3; i++)
		{
			size[i] = 0;
			while ((s[k] != ' ') && (s[k] != '\0'))
			{
				size[i] = size[i] * 10 + s[k] - 0x30;
				k++;
			}
			k++;

		};

		n = size[0];
		nnz = size[2];

		unsigned count(0);
		unsigned temp = 2 * nnz - n;

		AV.resize(temp);
		ANR.resize(temp);
		ANC.resize(temp);

		k = 0;
		while ((k < nnz) && (!data.eof()))
		{
			double val;
			unsigned val_r, val_c;
			data >> val_r >> val_c >> val;
			AV[count] = val;
			ANR[count] = val_r - 1;
			ANC[count] = val_c - 1;
			k++;
			count++;
			if (val_r != val_c)
			{
				AV[count] = val;
				ANR[count] = val_c - 1;
				ANC[count] = val_r - 1;
				count++;
			}
		}
		data.close();
		nnz = temp;
	}
};

double ScalProd(const vector<double> & a, const vector<double> & b, const size_t & StartIterr, const size_t & EndIterr);

int main()
{
	setlocale(LC_ALL, "RUSSIAN");
	sparsematrix S;
	S.ReadFromFile("ex5.mtx");
	vector<double> b(S.n);
	for (int i = 0; i < S.nnz; i++)
	b[S.ANR[i]] += S.AV[i];

	double NormR(0), NormB(0);
	for (int i = 0; i < b.size(); i++)
		NormB += b[i] * b[i];
	NormB = sqrt(NormB);

	double t1 = omp_get_wtime();
	std::cout << "Время запуска" << t1 << std::endl;

	double _ALPHA_CHISL_(0), _ALPHA_ZNAM_(0), _BETA_CHISL_(0), _BETA_ZNAM_(0);
	vector<double> x(S.n, 0.0), r(S.n), z(S.n), tmp(S.n);
	vector<double > r_new(r.size());
	int k = 0;

	omp_set_num_threads(1);
	
	int ii = 0;

#pragma omp parallel
	{
		double ALPHA(0), BETA(0);
		int num = omp_get_thread_num();
		int count = omp_get_num_threads();
		int tmp_num(num);
		vector<double> xx(S.n, 0.0);
		double _normR_(0);

#pragma omp for
		for (ii = 0; ii < S.n; ii++)
			z[ii] = r[ii] = b[ii];

		do
		{
#pragma omp for
			for (ii = 0; ii < tmp.size(); ii++)
			{
				tmp[ii] = 0.0;
			}

			for (int i = 0; i < tmp.size(); i++)
			{
				xx[i] = 0.0;
			}

			for (int i = num*(S.nnz) / count; i < (num + 1)*(S.nnz) / count; i++)
			{
				xx[S.ANR[i]] += S.AV[i] * z[S.ANC[i]];
			}

			for (int j = 0; j < count; j++)
			{
				tmp_num = (num + j) % count;
				for (int i = tmp_num *S.n / count; i < (tmp_num + 1)*S.n / count; i++)
					tmp[i] += xx[i];

			}
#pragma omp barrier 

			double _alpha_chisl_ = ScalProd(r, r, num*(S.n) / count, (num + 1)*(S.n) / count);
			double _alpha_znam_ = ScalProd(tmp, z, num*(S.n) / count, (num + 1)*(S.n) / count);

#pragma omp single
			{
				_ALPHA_CHISL_ = 0;
				_ALPHA_ZNAM_ = 0;
				_BETA_CHISL_ = 0;
				_BETA_ZNAM_ = 0;
				NormR = 0;
			}

 
#pragma omp atomic
			_ALPHA_CHISL_ += _alpha_chisl_;

#pragma omp atomic
			_ALPHA_ZNAM_ += _alpha_znam_;
#pragma omp barrier
			ALPHA = _ALPHA_CHISL_ / _ALPHA_ZNAM_;

#pragma omp for
			for (ii = 0; ii < x.size(); ii++)
				x[ii] += ALPHA * z[ii];

#pragma omp for
			for (ii = 0; ii < r_new.size(); ii++)
				r_new[ii] = r[ii] - ALPHA * tmp[ii];

			double _beta_chisl_ = ScalProd(r_new, r_new, num*(S.n) / count, (num + 1)*(S.n) / count);
			double _beta_znam_ = ScalProd(r, r, num*(S.n) / count, (num + 1)*(S.n) / count);

#pragma omp atomic
			_BETA_CHISL_ += _beta_chisl_;

#pragma omp atomic
			_BETA_ZNAM_ += _beta_znam_;
#pragma omp barrier

			BETA = _BETA_CHISL_ / _BETA_ZNAM_;

#pragma omp for
			for (ii = 0; ii < r.size(); ii++)
				r[ii] = r_new[ii];

#pragma omp for
			for (ii = 0; ii < z.size(); ii++)
				z[ii] = r[ii] + BETA * z[ii];

			_normR_ = 0;
#pragma omp for
			for (ii = 0; ii < r.size(); ii++)
				_normR_ += r[ii] * r[ii];

#pragma omp atomic
			NormR += _normR_;
#pragma omp barrier
#pragma omp single
			{
				NormR = sqrt(NormR);
				//if (!(k % 100))std::cout << "k = " << k << " NormR = " << NormR << endl;
				k++;
			}

		} while (NormR / NormB > 1e-6);

	}
	for (const auto s : x)
		std::cout << s << endl;


	double t2 = omp_get_wtime();

	for (auto &i : x)
		i -= 1;
	double normx = ScalProd(x, x, 0, x.size());
	normx = sqrt(normx);

	std::cout << "Времени затрачено: " << t2 - t1 << endl;
	std::cout << "Число итераций " << k << endl;
	std::cout << "Норма матрицы решений  " << normx << endl;
	std::system("pause");
	return 0;
}

double ScalProd(const vector<double>& a, const vector<double>& b, const size_t &StartIterr, const size_t& EndIterr)
{
	double tmp(0);
	for (size_t i = StartIterr; i < EndIterr; i++)
		tmp += a[i] * b[i];
	return tmp;
}
