#include "email.hpp"

namespace bdap {

    struct Accuracy {
        int n = 0;
        int correct = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            ++n;
            correct += static_cast<int>(lab == pred);
        }

        double get_accuracy() const { return static_cast<double>(correct) / n; }
        double get_error() const { return 1.0 - get_accuracy(); }

        double get_score() const { return get_accuracy(); }
    };

    // TODO add your own metrics below here

     struct Recall {
        int true_pos = 0;
        int false_neg = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
                
            if (lab && pred)
                ++true_pos;
            else if(lab && !pred)
                ++false_neg;
        }
                
        double get_recall() const { return (true_pos + 0.0) / (true_pos + false_neg); }
        double get_error() const { return 1.0 - get_recall(); }

        double get_score() const { return get_recall(); }
    };

     struct Precision {
        int true_pos = 0;
        int false_pos = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
                
            if (lab && pred)
                ++true_pos;
            else if(!lab && pred)
                ++false_pos;
        }
                
        double get_precision() const { return (true_pos + 0.0)/ (true_pos + false_pos); }
        double get_error() const { return 1.0 - get_precision(); }

        double get_score() const { return get_precision(); }
    };

    struct F1Score {
        int true_pos = 0;
        int false_neg = 0;
        int false_pos = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
                
            if (lab && pred)
                ++true_pos;
            else if(!lab && pred)
                ++false_pos;
            else if(lab && !pred)
                ++false_neg;
        }
                
        //double get_precision() const { return (true_pos + 0.0) / (true_pos + false_pos); }
        //double get_recall() const { return (true_pos + 0.0)/ (true_pos + false_neg); }
        //double precision = get_precision();
        //double recall = get_recall();
        double get_F1Score() const { return (2*((true_pos + 0.0) / (true_pos + false_pos))*((true_pos + 0.0)/ (true_pos + false_neg))) / (((true_pos + 0.0) / (true_pos + false_pos)) + ((true_pos + 0.0)/ (true_pos + false_neg))); }
        double get_error() const { return 1.0 - get_F1Score(); }

        double get_score() const { return get_F1Score(); }
    };

    struct ConfusionMatrix {

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, Email& emails) {

        }

    };


} // namespace bdap
