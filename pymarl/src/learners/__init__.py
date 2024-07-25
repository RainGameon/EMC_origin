from .q_learner import QLearner
from .qatten_learner import QattenLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qtran_transformation_learner import QLearner as QTranTRANSFORMATIONLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

from .fast_q_learner import fast_QLearner

from .qplex_curiosity_vdn_learner import QPLEX_curiosity_vdn_Learner

from .qplex_curiosity_vdn_learner_COIN import QPLEX_curiosity_vdn_Learner_COIN
from .qplex_curiosity_vdn_learner_inf import QPLEX_curiosity_vdn_Learner_inf
from .qplex_curiosity_vdn_learner_ps import QPLEX_curiosity_vdn_Learner_ps



REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qtran_transformation_learner"] = QTranTRANSFORMATIONLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner


REGISTRY["fast_QLearner"] = fast_QLearner
REGISTRY['qplex_curiosity_vdn_learner']=QPLEX_curiosity_vdn_Learner
REGISTRY['qplex_curiosity_vdn_learner_COIN']=QPLEX_curiosity_vdn_Learner_COIN
REGISTRY['qplex_curiosity_vdn_learner_inf']=QPLEX_curiosity_vdn_Learner_inf
REGISTRY['qplex_curiosity_vdn_learner_ps']=QPLEX_curiosity_vdn_Learner_ps


