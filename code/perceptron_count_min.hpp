#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    int ngram_;
    int seed_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    int num_hashes_;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> counts_;

public:
    /** Do not change the signature of the constructor! */
    PerceptronCountMin(int ngram, int num_hashes, int log_num_buckets,
                       double learning_rate)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , num_hashes_(num_hashes)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0xa738cc)
       
    {
        size_t num_buckets_ = 1 << log_num_buckets_;

        // create a weights matrix and initialize to 0
        weights_.resize(num_hashes_);
        for (int i=0; i<num_hashes_; ++i){
            weights_[i].resize(num_buckets_,0);
        }

        // create a counts matrix and initialize to 0
        counts_.resize(num_hashes_);
        for (int i=0; i<num_hashes_; ++i){
            counts_[i].resize(num_buckets_,0);
        }


    }

    void update_(const Email& email) {
        // TODO implement this
        //create an instance of the EmailIter class with n-gram size
        EmailIter emailiter(email, ngram_);
        size_t num_buckets_ = 1 << log_num_buckets_;

        
        //label email 1 if it's spam or -1 if it's ham 
        int label;
        if (email.is_spam()) {
            label = 1;
        } else {
            label = -1;
        }

        double prediction=activate(predict_(email));
        double error=label-prediction;

        for(int i=0; i<num_hashes_; ++i){
            EmailIter emailiter(email, ngram_);
            bias_+=learning_rate_*error;

            while(emailiter){
                std::string_view ngram_= emailiter.next();
                size_t bucket = hash(ngram_, seed_ + i) % num_buckets_;
                counts_[i][bucket]++;
                weights_[i][bucket]+=learning_rate_*error*counts_[i][bucket];
            }
        }
                
    }

    double predict_(const Email& email) const {
        // TODO implement this
         //create an instance of the EmailIter class with n-gram size
        EmailIter emailiter(email, ngram_);
        size_t num_buckets_ = 1 << log_num_buckets_;

        std::vector<int> weight(num_hashes_);
        double prediction = 0.0;

        while (emailiter) {
            std::string_view ngram_= emailiter.next();
            
            for (int i = 0; i < num_hashes_; ++i) {
                size_t bucket = hash(ngram_, seed_ + i) % num_buckets_;
                weight[i] = weights_[i][bucket];
            }  

            
            // Calculate the dot product using only the hash function with the medianweight
            std::sort(weight.begin(), weight.end());
            int medianWeight;

            if (num_hashes_ % 2 == 0) {
                // If even, average the two middle values
                int mid1= weight[num_hashes_ / 2 - 1];
                int mid2 = weight[num_hashes_ / 2];
                medianWeight = (mid1 + mid2) / 2;
            } else {
                // If odd, take the middle value
                medianWeight = weight[num_hashes_ / 2];
            }
            
            for (int i = 0; i < num_hashes_; ++i) {
                size_t bucket = hash(ngram_, seed_ + i) % num_buckets_;
                prediction += medianWeight * counts_[i][bucket];
            } 
        
        }

        prediction+=bias_;


        return prediction;
    }

private:

    // activation function
    double activate(double value) const {
        if ( value >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

};

} // namespace bdap
