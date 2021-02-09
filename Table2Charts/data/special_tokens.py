from .config import DataConfig
from .token import Token, TokenType, AggFunc, AnaType, GroupingOp
from .util import get_embeddings


class SpecialTokens:
    ANA_TOKEN = Token(TokenType.ANA)
    PAD_TOKEN = Token(TokenType.PAD)
    SEP_TOKEN = Token(TokenType.SEP)
    GRP_OP_TOKENS = [Token(TokenType.GRP, grp_op=GroupingOp.Cluster),
                     Token(TokenType.GRP, grp_op=GroupingOp.Stack)]

    @classmethod
    def get_grp_token(cls, grp_op: GroupingOp):
        if grp_op is GroupingOp.Cluster:
            return cls.GRP_OP_TOKENS[0]
        else:  # Stack
            return cls.GRP_OP_TOKENS[1]

    @staticmethod
    def get_ana_token(ana_type: AnaType):
        return Token(TokenType.ANA, ana_type=ana_type)
