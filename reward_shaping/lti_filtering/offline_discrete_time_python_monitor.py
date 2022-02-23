from rtamt.operation.stl.discrete_time.offline.constant_operation import ConstantOperation
from rtamt.spec.stl.discrete_time.visitor import STLVisitor
from rtamt.operation.arithmetic.discrete_time.offline.addition_operation import AdditionOperation
from rtamt.operation.arithmetic.discrete_time.offline.multiplication_operation import MultiplicationOperation
from rtamt.operation.arithmetic.discrete_time.offline.subtraction_operation import SubtractionOperation
from rtamt.operation.arithmetic.discrete_time.offline.division_operation import DivisionOperation
from rtamt.operation.stl.discrete_time.offline.and_operation import AndOperation
from rtamt.operation.stl.discrete_time.offline.or_operation import OrOperation
from rtamt.operation.arithmetic.discrete_time.offline.abs_operation import AbsOperation
from rtamt.operation.arithmetic.discrete_time.offline.sqrt_operation import SqrtOperation
from rtamt.operation.arithmetic.discrete_time.offline.exp_operation import ExpOperation
from rtamt.operation.arithmetic.discrete_time.offline.pow_operation import PowOperation
from rtamt.operation.stl.discrete_time.offline.not_operation import NotOperation
from rtamt.operation.stl.discrete_time.offline.always_operation import AlwaysOperation
# custom re-implemented operators
from reward_shaping.lti_filtering.filtering_operations import PredicateOperation, EventuallyOperation


class MTLFilteringOfflineDiscreteTimePythonMonitor(STLVisitor):
    def __init__(self):
        self.node_monitor_dict = dict()

    def generate(self, node):
        self.visit(node, [])
        return self.node_monitor_dict

    def visitPredicate(self, node, args):
        monitor = PredicateOperation(node.operator)
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitVariable(self, node, args):
        pass

    def visitAbs(self, node, args):
        monitor = AbsOperation()
        self.node_monitor_dict[node.name] = monitor
        self.visit(node.children[0], args)

    def visitSqrt(self, node, args):
        monitor = SqrtOperation()
        self.node_monitor_dict[node.name] = monitor
        self.visit(node.children[0], args)

    def visitExp(self, node, args):
        monitor = ExpOperation()
        self.node_monitor_dict[node.name] = monitor
        self.visit(node.children[0], args)

    def visitPow(self, node, args):
        monitor = PowOperation()
        self.node_monitor_dict[node.name] = monitor
        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitAddition(self, node, args):
        monitor = AdditionOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitSubtraction(self, node, args):
        monitor = SubtractionOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitMultiplication(self, node, args):
        monitor = MultiplicationOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitDivision(self, node, args):
        monitor = DivisionOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitNot(self, node, args):
        monitor = NotOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)

    def visitAnd(self, node, args):
        monitor = AndOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitOr(self, node, args):
        monitor = OrOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)
        self.visit(node.children[1], args)

    def visitImplies(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitIff(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitXor(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitEventually(self, node, args):
        monitor = EventuallyOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)

    def visitAlways(self, node, args):
        monitor = AlwaysOperation()
        self.node_monitor_dict[node.name] = monitor

        self.visit(node.children[0], args)

    def visitUntil(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitOnce(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitHistorically(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitSince(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitRise(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitFall(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitConstant(self, node, args):
        # what is this?
        monitor = ConstantOperation(node.val)
        self.node_monitor_dict[node.name] = monitor

    def visitPrevious(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitNext(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedPrecedes(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedOnce(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedHistorically(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedSince(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedAlways(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedEventually(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitTimedUntil(self, node, args):
        raise NotImplementedError("operator not implemented")

    def visitDefault(self, node, args):
        pass


