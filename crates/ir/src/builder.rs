//! Fail-closed builder for graph IR values.

use crate::{DType, Graph, IrError, Node, NodeId, Op, Shape, TensorData, TensorType};

/// Builder for an arena-backed graph.
#[derive(Debug, Clone, Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl GraphBuilder {
    /// Creates an empty graph builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an input node and returns its id.
    ///
    /// # Errors
    ///
    /// This method currently has no failing input case, but returns
    /// `Result` to keep construction fail-closed as validation rules grow.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: Shape,
    ) -> Result<NodeId, IrError> {
        let ty = TensorType::new(dtype, shape);
        let id = self.push(Op::Input {
            name: name.into(),
            ty,
        });
        self.inputs.push(id);
        Ok(id)
    }

    /// Appends a constant node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::ShapeOverflow`] if the shape element count overflows,
    /// or [`IrError::ConstDataShapeMismatch`] if the payload length does not
    /// match the shape.
    pub fn constant(&mut self, data: TensorData, shape: Shape) -> Result<NodeId, IrError> {
        validate_const_data(&data, &shape)?;
        Ok(self.push(Op::Const { data, shape }))
    }

    /// Appends an elementwise add node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if either operand is outside this
    /// builder's current arena.
    pub fn add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, IrError> {
        self.binary(left, right, Op::Add)
    }

    /// Appends an elementwise subtract node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if either operand is outside this
    /// builder's current arena.
    pub fn sub(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, IrError> {
        self.binary(left, right, Op::Sub)
    }

    /// Appends an elementwise multiply node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if either operand is outside this
    /// builder's current arena.
    pub fn mul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, IrError> {
        self.binary(left, right, Op::Mul)
    }

    /// Appends an elementwise negate node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `input` is outside this builder's
    /// current arena.
    pub fn neg(&mut self, input: NodeId) -> Result<NodeId, IrError> {
        self.unary(input, Op::Neg)
    }

    /// Appends an elementwise `ReLU` node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `input` is outside this builder's
    /// current arena.
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, IrError> {
        self.unary(input, Op::Relu)
    }

    /// Appends a two-dimensional matmul node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if either operand is outside this
    /// builder's current arena.
    pub fn matmul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, IrError> {
        self.binary(left, right, Op::Matmul)
    }

    /// Appends a sum-reduction node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `input` is outside this builder's
    /// current arena.
    pub fn sum(
        &mut self,
        input: NodeId,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Result<NodeId, IrError> {
        self.validate_operand(input)?;
        Ok(self.push(Op::Sum {
            input,
            axes,
            keepdim,
        }))
    }

    /// Appends a reshape node and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `input` is outside this builder's
    /// current arena.
    pub fn reshape(&mut self, input: NodeId, new_shape: Shape) -> Result<NodeId, IrError> {
        self.validate_operand(input)?;
        Ok(self.push(Op::Reshape { input, new_shape }))
    }

    /// Marks a node as a graph output.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `id` is outside this builder's
    /// current arena.
    pub fn mark_output(&mut self, id: NodeId) -> Result<(), IrError> {
        self.validate_operand(id)?;
        self.outputs.push(id);
        Ok(())
    }

    /// Builds a validated graph.
    ///
    /// # Errors
    ///
    /// Returns the first structural validation error from
    /// [`Graph::validate_structure`].
    pub fn build(self) -> Result<Graph, IrError> {
        let graph = Graph::new(self.nodes, self.inputs, self.outputs);
        graph.validate_structure()?;
        Ok(graph)
    }

    fn binary(
        &mut self,
        left: NodeId,
        right: NodeId,
        make_op: impl FnOnce(NodeId, NodeId) -> Op,
    ) -> Result<NodeId, IrError> {
        self.validate_operand(left)?;
        self.validate_operand(right)?;
        Ok(self.push(make_op(left, right)))
    }

    fn unary(
        &mut self,
        input: NodeId,
        make_op: impl FnOnce(NodeId) -> Op,
    ) -> Result<NodeId, IrError> {
        self.validate_operand(input)?;
        Ok(self.push(make_op(input)))
    }

    fn validate_operand(&self, id: NodeId) -> Result<(), IrError> {
        if id.index() < self.nodes.len() {
            Ok(())
        } else {
            Err(IrError::InvalidNodeId {
                id,
                num_nodes: self.nodes.len(),
            })
        }
    }

    fn push(&mut self, op: Op) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(Node::new(op));
        id
    }
}

fn validate_const_data(data: &TensorData, shape: &Shape) -> Result<(), IrError> {
    let expected = shape.numel()?;
    let got = data.len();
    if got == expected {
        Ok(())
    } else {
        Err(IrError::ConstDataShapeMismatch { expected, got })
    }
}
