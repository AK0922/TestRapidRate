from src.irateyourate.processors.processor import Processor
from src.irateyourate.utils import log_helper, doc2vec_helper, review_file_helper, ml_helper
from src.irateyourate.utils.options import Options

log = log_helper.get_logger("AmazonReviewProcessor")


class AmazonReviewProcessor(Processor):

    def process(self):

        log.info("Processing begun")

        log.info("Reading input file " + Options.options.input_file_path)
        tagged_reviews, rating_dict = review_file_helper.parse_review_file()

        log.info("Building Doc2Vec model")
        doc2vec_model = doc2vec_helper.init_doc2vec_model(tagged_reviews)
        doc2vec_helper.train_doc2vec_model(doc2vec_model, tagged_reviews)
        log.info("Doc2Vec model successfully trained")

        log.info("Saving Doc2Vec model to disk " + Options.options.doc2vec_model_path)
        doc2vec_model.save(Options.options.doc2vec_model_path)

        log.info("Extracting train/test parameters")
        x_docvecs, y_scores = ml_helper.extract_training_parameters(doc2vec_model, rating_dict)

        log.info("Training Linear Regression model")
        lr_model = ml_helper.train_linear_model(x_docvecs, y_scores)
        log.info("Training Support Vector Regression model")
        svm_model = ml_helper.train_svm(x_docvecs, y_scores)
        log.info("ML Model training done")

        log.info("Persisting ML models to disk " + Options.options.ml_model_path)
        ml_helper.persist_model_to_disk(lr_model, Options.options.ml_model_path + ".lr")
        ml_helper.persist_model_to_disk(svm_model, Options.options.ml_model_path + ".svm")

        log.info("Execution complete")
