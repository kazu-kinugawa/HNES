#pragma once

#include <iostream>
#include <fstream>
#include "Matrix.hpp"

/* https://www.quora.com/How-can-I-declare-an-unordered-set-of-pair-of-int-int-in-C++11 */
// for unordered_map< std::pair<int, int>, int >
template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
  template<typename S, typename T> struct hash<pair<S, T>>
  {
    inline size_t operator()(const pair<S, T> & v) const
    {
      size_t seed = 0;
      ::hash_combine(seed, v.first);
      ::hash_combine(seed, v.second);
      return seed;
    }
  };
}
/* https://www.quora.com/How-can-I-declare-an-unordered-set-of-pair-of-int-int-in-C++11 */

class ROUGE{
  public:
    Vec3D rouge1;// recall, precision, f-score
    Vec3D rouge2;// recall, precision, f-score
    Vec3D rougeL;// recall, precision, f-score

    void operator = (const ROUGE& rouge){
      this->rouge1 = rouge.rouge1;
      this->rouge2 = rouge.rouge2;
      this->rougeL = rouge.rougeL;
    }
    void operator += (const ROUGE& rouge){
      this->rouge1 += rouge.rouge1;
      this->rouge2 += rouge.rouge2;
      this->rougeL += rouge.rougeL;
    }
    void operator -= (const ROUGE& rouge){
      this->rouge1 -= rouge.rouge1;
      this->rouge2 -= rouge.rouge2;
      this->rougeL -= rouge.rougeL;
    }
    void operator *= (const double scalor){
      this->rouge1 *= scalor;
      this->rouge2 *= scalor;
      this->rougeL *= scalor;
    }
    void operator /= (const double scalor){
      this->rouge1 /= scalor;
      this->rouge2 /= scalor;
      this->rougeL /= scalor;
    }
    void init(){
      this->rouge1.setZero();
      this->rouge2.setZero();
      this->rougeL.setZero();
    }
    ROUGE(){
      this->init();
    }
    ROUGE(const ROUGE& rouge){
      this->rouge1 = rouge.rouge1;
      this->rouge2 = rouge.rouge2;
      this->rougeL = rouge.rougeL;
    }

    void print(){
      std::cout << "ROUGE-1 : Recall =   " << this->rouge1.coeffRef(0,0) << std::endl;
      std::cout << "ROUGE-1 : Precison = " << this->rouge1.coeffRef(1,0) << std::endl;
      std::cout << "ROUGE-1 : F-score =  " << this->rouge1.coeffRef(2,0) << std::endl;
      std::cout << "ROUGE-2 : Recall =   " << this->rouge2.coeffRef(0,0) << std::endl;
      std::cout << "ROUGE-2 : Precison = " << this->rouge2.coeffRef(1,0) << std::endl;
      std::cout << "ROUGE-2 : F-score =  " << this->rouge2.coeffRef(2,0) << std::endl;
      std::cout << "ROUGE-L : Recall =   " << this->rougeL.coeffRef(0,0) << std::endl;
      std::cout << "ROUGE-L : Precison = " << this->rougeL.coeffRef(1,0) << std::endl;
      std::cout << "ROUGE-L : F-score =  " << this->rougeL.coeffRef(2,0) << std::endl;
    }
    void save(std::ofstream& fout){
      fout << this->rouge1.coeffRef(0,0) << "," << this->rouge1.coeffRef(1,0) << "," << this->rouge1.coeffRef(2,0) << ",";
      fout << this->rouge2.coeffRef(0,0) << "," << this->rouge2.coeffRef(1,0) << "," << this->rouge2.coeffRef(2,0) << ",";
      fout << this->rougeL.coeffRef(0,0) << "," << this->rougeL.coeffRef(1,0) << "," << this->rougeL.coeffRef(2,0) << std::endl;
    }
    void lcs(const std::vector<int>& X, const std::vector<int>& Y, std::vector<int>& res){
      const size_t m = X.size();
      const size_t n = Y.size();
      MatI L = MatI::Zero(m+1,n+1);

      /* Following steps build L[m+1][n+1] in bottom up fashion. Note
         that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] */
      for (size_t i = 0; i <= m; ++i){
        for (size_t j = 0; j <= n; ++j){
          if (i == 0 || j == 0){
            L.coeffRef(i,j) = 0;
          }
          else if (X[i-1] == Y[j-1]){
            L.coeffRef(i,j) = L.coeffRef(i-1,j-1) + 1;
          }
          else{
            L.coeffRef(i,j) = std::max(L.coeffRef(i-1,j), L.coeffRef(i,j-1));
          }
        }
      }

      // Following code is used to print LCS
      int index = L.coeffRef(m,n);

      // Create a character array to store the lcs string
      //char lcs[index+1];
      //lcs[index] = '\0'; // Set the terminating character
      res.resize(index);

      // Start from the right-most-bottom-most corner and
      // one by one store characters in lcs[]
      int i = m, j = n;
      while (i > 0 && j > 0){
        // If current character in X[] and Y are same, then
        // current character is part of LCS
        if (X[i-1] == Y[j-1]){
          res[index-1] = X[i-1]; // Put current character in result
          --i;
          --j;
          --index;     // reduce values of i, j and index
        }

        // If not same, then find the larger of two and
        // go in the direction of larger value
        else if (L.coeffRef(i-1,j) > L.coeffRef(i,j-1)){
          --i;
        }
        else{
          --j;
        }
      }
      std::sort(res.begin(), res.end());
      // Print the lcs
      /*
        std::cout << std::endl;
        std::cout << "LCS is " << std::endl;
        for(size_t i = 0, i_end = res.size(); i < i_end; ++i){
        std::cout << res[i] << " ";
        }
        std::cout << std::endl;
      */
    }
};

// https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
/*
std::size_t operator()(std::vector<uint32_t> const& vec) const {
  std::size_t seed = vec.size();
  for(auto& i : vec) {
    seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
*/
