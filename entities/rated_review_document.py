import json
from gensim import utils
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument


class RatedReviewDocument(TaggedLineDocument):

    def __init__(self, source):
        super(RatedReviewDocument, self).__init__(source)

    def __iter__(self):
        try:
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield \
                    TaggedDocument(
                        utils.to_unicode(
                            self.get_review_content_from_line(line)
                        ).split(),
                        [item_no]
                    )
        except AttributeError:
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield \
                        TaggedDocument(
                            utils.to_unicode(
                                self.get_review_content_from_line(line)
                            ).split(),
                            [item_no]
                        )

    @staticmethod
    def get_review_content_from_line(line):
        review_dict = json.loads(line)
        return review_dict['reviewText']
