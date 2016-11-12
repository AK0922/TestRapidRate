from src.irateyourate.processors.processor import Processor
from src.irateyourate.utils import log_helper, doc2vec_helper, review_file_helper, ml_helper
from src.irateyourate.utils.options import Options

log = log_helper.get_logger("AmazonReviewProcessor")


class AmazonReviewProcessor(Processor):

    def process(self):

        log.info("Processing begun")

        # Use generator to read the input file and create docvecs for each
        log.info("Reading input file " + Options.options.input_file_path)
        tagged_reviews, rating_dict = review_file_helper.parse_review_file()

        log.info("Building Doc2Vec model")
        doc2vec_model = doc2vec_helper.init_doc2vec_model(tagged_reviews)
        doc2vec_helper.train_doc2vec_model(doc2vec_model, tagged_reviews)
        log.info("Doc2Vec model successfully trained")

        log.info("Saving Doc2Vec model to disk " + Options.options.doc2vec_model_path)
        doc2vec_model.save(Options.options.doc2vec_model_path)

        log.info("Training ML model")
        x_docvecs, y_scores = ml_helper.extract_training_parameters(doc2vec_model, rating_dict)
        ml_model = ml_helper.train_linear_model(x_docvecs, y_scores)
        log.info("ML model training done")

        log.info("Persisting ML model to disk " + Options.options.ml_model_path)
        ml_helper.persist_model_to_disk(ml_model)

        log.info("Execution complete")
