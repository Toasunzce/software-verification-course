class DocMindError(Exception):
    """Base application error."""


class DocumentNotFoundError(DocMindError):
    pass


class InvalidTransitionError(DocMindError):
    pass


class DocumentNotReadyError(DocMindError):
    pass


class UnsupportedFileTypeError(DocMindError):
    pass


class LLMError(DocMindError):
    pass