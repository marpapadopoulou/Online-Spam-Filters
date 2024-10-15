#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    std::vector<int> counts_;

    int seed_;

public:
    /** Do not change the signature of the constructor! */
    PerceptronFeatureHashing(int ngram, int log_num_buckets, double learning_rate)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0xa738cc)
        , counts_(1 << log_num_buckets_, 0) // size 2^log_num_buckets_ and initialize all elemets to 0
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email) {
        // TODO implement this
        //create an instance of the EmailIter class with n-gram size 
        EmailIter emailiter(email, ngram_); 

        //label email 1 if it's spam or -1 if it's ham 
        int label;
        if (email.is_spam()) {
            label = 1;
        } else {
            label = -1;
        }

        double prediction=predict_(email);
        double error=label-activate(prediction);


        bias_ += learning_rate_ * error;


        while (emailiter) {
            std::string_view ngram = emailiter.next(); //iterate through the n grams and extract one at the time 
            size_t bucket = get_bucket(ngram ); // hash the n-grams to a bucket
            counts_[bucket]++; //increment the count of the bucket

            weights_[bucket] += learning_rate_ * error*counts_[bucket];
        }
    }
           
    double predict_(const Email& email) const {
        // TODO implement this
        EmailIter emailiter(email, ngram_);

        double prediction =0.0;
         while (emailiter) {
            std::string_view ngram = emailiter.next(); //iterate through the n grams and extract one at the time 
            size_t bucket = get_bucket(ngram ); // hash the n-grams to a bucket
            prediction+=weights_[bucket]*counts_[bucket];
        }

        prediction+=bias_;
        
        return prediction;
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        // TODO limit the range of the hash values here
         // first i get the number of buckets from their logarithmic representation using bit manipulation
        size_t num_buckets = 1 << log_num_buckets_;
        hash=hash%num_buckets; //apply modulo to ensure that the hash value falls into the range of available buckets -> prevent overflow
        return hash;
    }

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
