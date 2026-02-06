"""Parser for Carnot Deep Research output. Same format as deepscholar_base (intro.md + paper.csv)."""

from .deepscholar_base import DeepScholarBaseParser
from .parser_type import ParserType


class CarnotDeepResearchParser(DeepScholarBaseParser):
    """Carnot output uses the same intro.md + paper.csv format as deepscholar_base."""

    parser_type = ParserType.CARNOT_DEEP_RESEARCH
