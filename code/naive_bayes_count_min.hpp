#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
    int ngram_;
    int seed_;
    int num_spam=0;
    int num_ham=0;
    int num_hashes_;
    int log_num_buckets_;
    std::vector<std::vector<int>> cms_spam_;
    std::vector<std::vector<int>> cms_ham_;

public:
    NaiveBayesCountMin(int ngram, int num_hashes, int log_num_buckets)
        : BaseClf(-4 /* set appropriate threshold */)
        , ngram_(ngram)
        , seed_(0xfa4f8cc)
        , num_hashes_(num_hashes)
        , log_num_buckets_(log_num_buckets)
    {
        size_t num_buckets_ = 1 << log_num_buckets_;
        
        // create different cms matrixes for spam and ham classes and initialize to 0
        cms_spam_.resize(num_hashes_);
        cms_ham_.resize(num_hashes_);

        for (int i=0; i<num_hashes_; ++i){
            cms_spam_[i].resize(num_buckets_,0);
            cms_ham_[i].resize(num_buckets_,0);
        }

    }

    void update_(const Email &email) {
        // TODO implement this
        //create an instance of the EmailIter class with n-gram size
        EmailIter emailiter(email, ngram_);
        // get the number of buckets from their logarithmic representation using bit manipulation
        size_t num_buckets_ = 1 << log_num_buckets_;

        //calculate the spam/ham emails
         if(email.is_spam()){
            num_spam++;
            updateCountMinSketch(cms_spam_ , emailiter, num_buckets_);
        } else{
            num_ham++;
            updateCountMinSketch(cms_ham_ , emailiter, num_buckets_);
        }

    }
       

    double predict_(const Email& email) const {
        // TODO implement this
        int total_spam=0;
        int total_ham=0;

        //calculate P(S) and P(H)

        double log_prob_spam = log(static_cast<double>(num_spam)/(num_spam+num_ham));
        double log_prob_ham = log(static_cast<double>(num_ham)/(num_spam+num_ham));

        //calculate total count of ngrams in each class 
        total_spam=calculateTotalCount(cms_spam_);
        total_ham=calculateTotalCount(cms_ham_);


        //calculate the log-likelihood of each n-gram occurrence given spam or ham class
        double log_likelihood_spam = calculateLogLikelihood(email, cms_spam_);
        double log_likelihood_ham = calculateLogLikelihood(email, cms_ham_);

        //calculate P(S|text)
        double overall_prob= log_likelihood_spam +log_prob_spam - log_likelihood_ham - log_prob_ham;
        
        return overall_prob;
    }

private:
    // function to update the Count-Min Sketch matrix
    void updateCountMinSketch(std::vector<std::vector<int>>& cms, EmailIter& emailiter, size_t num_buckets) {
        while (emailiter) {
            std::string_view ngram = emailiter.next(); // iterate through the n grams and extract one at a time

            // each n-gram is hashed num_hashes_ different times
            for (int i = 0; i < num_hashes_; ++i) {
                size_t bucket = hash(ngram, seed_ + i) % num_buckets; // use % num_buckets to prevent overflow
                cms[i][bucket] += 1; 
            }
        }
    }

    int calculateTotalCount(const std::vector<std::vector<int>>& cms) const {
        int total_count = 0;

        // Iterate through all buckets in the Count-Min Sketch matrix
        for (const auto& row : cms) {
            for (int count : row) {
                total_count += count;
            }
        }

        return total_count;
    }

    double calculateLogLikelihood(const Email& email, const std::vector<std::vector<int>>& cms) const {
        double log_likelihood = 0.0;

        //calculate the log-likelihood of each n-gram occurrence given spam or ham
        EmailIter emailiter(email, ngram_); 
        while (emailiter) {
            std::string_view ngram = emailiter.next();
            size_t num_buckets_ = 1 << log_num_buckets_;
            std::vector<int> counts;

             // Iterate through hash functions and get the counts for the ngram
             for (int i = 0; i < num_hashes_; ++i) {
                size_t bucket = hash(ngram, seed_ + i) % num_buckets_;
                counts.push_back(cms[i][bucket]);
            }

            // get the minimum of the counts
            int min_count = *std::min_element(counts.begin(), counts.end());

            log_likelihood += log(static_cast<double>(min_count + 1) / (calculateTotalCount(cms)+1));
        }

        return log_likelihood;
    }

};

} // namespace bdap
