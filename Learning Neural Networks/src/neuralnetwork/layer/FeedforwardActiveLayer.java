package neuralnetwork.layer;

import java.util.Arrays;

import neuralnetwork.activation.ActivationSigmoid;

public class FeedforwardActiveLayer implements FeedforwardLayer {

	public enum Type {HIDDEN, OUTPUT};
	
	private ActivationSigmoid function;
	
	private double[] outputs;
	private double[] deltas;

	private double[][] weights;
	private double[][] accumulatedErrorDelta;
	
	public FeedforwardActiveLayer(Type type, int numNodes, int prevNumNodes) {
		this.function = ActivationSigmoid.getInstance();
		
		if (type == Type.HIDDEN) {
			this.outputs = new double [numNodes + 1];
			this.outputs [numNodes] = 1;
		}
		else {
			this.outputs = new double [numNodes];			
		}
		
		this.deltas = new double [numNodes];
		
		this.weights = new double [numNodes][prevNumNodes + 1];
		this.accumulatedErrorDelta = new double [numNodes][prevNumNodes + 1];
		
		randomizeWeights();
	}
	
	public void calculateOutputs(double[] input) {
		for (int i = 0; i < getNumNodes(); i++) {
			double sum = calculateWeightedSum(input, i);
			outputs [i] = function.activationFunction(sum);
		}
	}
	
	private double calculateWeightedSum(double[] input, int node) {
		double sum = 0;
		
		for (int i = 0; i < input.length; i++) {
			sum += input [i] * weights [node][i];
		}
		
		return sum;
	}
	
	private double getDelta(int node) {
		return deltas [node];
	}
	
	private double getNodeWeight(int node, int prevNode) {
		return weights [node][prevNode];
	}
	
	// output layer
	public void backpropagationNodeDeltas(double[] targets) {
		for (int i = 0; i < getNumNodes(); i++) {
			double error = -2 * (targets [i] - outputs [i]);
			deltas [i] = error * function.derivativeFunction(outputs [i]);
		}
	}
	
	// hidden layers
	public void backpropagationNodeDeltas(FeedforwardActiveLayer nextLayer) {
		for (int i = 0; i < getNumNodes(); i++) {
			deltas [i] = 0;
			
			for (int j = 0; j < nextLayer.getNumNodes(); j++) {
				double delta = nextLayer.getNodeWeight(j, i) * nextLayer.getDelta(j);
				deltas [i] += delta;
			}
			
			deltas [i] *= function.derivativeFunction(outputs [i]);
		}
	}
	
	public void incrementErrorDelta(FeedforwardLayer prevLayer) {
		for (int i = 0; i < getNumNodes(); i++) {
			for (int j = 0; j < getPrevNumNodes(); j++) {
				double increment = deltas [i] * prevLayer.getOutputs() [j];
				accumulatedErrorDelta [i][j] += increment;
			}
		}
	}
	
	public void updateWeightsBatch(double learningRate) {
		for (int i = 0; i < getNumNodes(); i++) {
			for (int j = 0; j < getPrevNumNodes(); j++) {
				weights [i][j] -= learningRate * accumulatedErrorDelta [i][j]; 					
			}
		}
	}
	
	@Override
	public double[] getOutputs() {
		return outputs;
	}
	
	@Override
	public int getNumNodes() {
		return weights.length;
	}
	
	int getPrevNumNodes() {
		return weights [0].length;
	}

	void randomizeWeights() {
		for (int row = 0; row < getNumNodes(); row++) {
			for (int col = 0; col < getPrevNumNodes(); col++) {
				weights [row][col] = -0.5 + (Math.random() * 1);
			}
		}
	}

	public void resetAccumulatedErrorDelta() {
		for (int row = 0; row < getNumNodes(); row++) {
			for (int col = 0; col < getPrevNumNodes(); col++) {
				accumulatedErrorDelta [row][col] = 0;
			}
		}
	}

	@Override
	public String toString() {
		String s = "[";
		
		for (int i = 0; i < weights.length; i++) {
			s += Arrays.toString(weights [i]);
			
			if (i != weights.length - 1) {
				s += "\n";
			}
		}
		
		s += "]";
		
		return s;
	}
	
}
