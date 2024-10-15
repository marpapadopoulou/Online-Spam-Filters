#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int seed_;
    int ngram_;
    int log_num_buckets_;
    int num_spam=0;
    int num_ham=0;
    std::vector<int> counts_;
    std::vector<int> spam_counts_;
    std::vector<int> ham_counts_;

public:
    /** Do not change the signature of the constructor! */
    NaiveBayesFeatureHashing(int ngram, int log_num_buckets)
        : BaseClf(-1 /* set appropriate threshold */)
        , seed_(0xfa4f8cc)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , spam_counts_(1 << log_num_buckets_, 1) // size 2^log_num_buckets_ and initialize all elemets to 1
        , ham_counts_(1 << log_num_buckets_, 1)
    {}

    void update_(const Email &email) {
        // TODO implement this
        //create an instance of the EmailIter class with n-gram size 
        EmailIter emailiter(email, ngram_); 

        //calculate the spam/ham emails
         if(email.is_spam()){
            num_spam++;
        } else{
            num_ham++;
        }

        while (emailiter) {
            std::string_view ngram = emailiter.next(); //iterate through the n grams and extract one at the time 
            size_t bucket = get_bucket (ngram , email.is_spam()); // hash the n-grams to a bucket
        
            if (email.is_spam()) {
            spam_counts_[bucket]++; // Increment the count of the bucket for spam
            } else {
            ham_counts_[bucket]++;  // Increment the count of the bucket for ham
            }
        }
    }

    double predict_(const Email& email) const {
        // TODO implement this
       
        int total_spam=0; //total spam words-ngrams
        int total_ham=0;  //total ham words-ngrams


        //calculate P(S) and P(H)

        double log_prob_spam = log(static_cast<double>(num_spam)/(num_spam+num_ham));
        double log_prob_ham = log(static_cast<double>(num_ham)/(num_spam+num_ham));

        //calculate P(W|S) and P(W|H)

        //add 1 for Laplace smoothing
        for(const int& count  :spam_counts_){
            total_spam = total_spam +count + 1; //total counts of ngrams in spam class
        }

        for(const int& count : ham_counts_){
            total_ham = total_ham + count + 1; //total counts of ngrams in ham class
        }

        double log_like_prob_spam=0.0;
        double log_like_prob_ham=0.0;

        //calculate the log-likelihood of each n-gram occurrence given spam or ham class 
        //add 1 to the numerator and 2 to the denominator (we have 2 classes -. each feature has 2 possible outcomes) for Laplace smoothing
        EmailIter emailiter(email, ngram_); 
        while (emailiter) {
            std::string_view ngram = emailiter.next();
            size_t bucket = get_bucket(ngram, email.is_spam());

            log_like_prob_spam += log(static_cast<double>(spam_counts_[bucket] + 1) / (total_spam + 2 ));
            log_like_prob_ham +=log(static_cast<double>(ham_counts_[bucket] + 1) / (total_ham + 2 ));
        }

        //calculate P(S|text)
        double overall_prob= log_like_prob_spam+log_prob_spam-log_like_prob_ham-log_prob_ham;

        //std::cout << "Overall Probability: " << overall_prob << std::endl;

        return overall_prob;
    }

private:
    size_t get_bucket(std::string_view ngram, int is_spam) const {
        return get_bucket(hash(ngram, seed_), is_spam);
    }

    size_t get_bucket(size_t hash, int is_spam) const {
        // TODO limit the range of the hash values here
        // first i get the number of buckets from their logarithmic representation using bit manipulation
        size_t num_buckets = 1 << log_num_buckets_;
        hash=hash%num_buckets; //apply modulo to ensure that the hash value falls into the range of available buckets -> prevent overflow
        return hash;
    }


};
 // namespace bdap
}