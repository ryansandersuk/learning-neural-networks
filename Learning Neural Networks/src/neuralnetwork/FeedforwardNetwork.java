package neuralnetwork;

import neuralnetwork.error.ErrorCalculation;
import neuralnetwork.layer.FeedforwardActiveLayer;
import neuralnetwork.layer.FeedforwardInputLayer;

public class FeedforwardNetwork {

	private double learningRate;
	
	private FeedforwardInputLayer inputLayer;
	private FeedforwardActiveLayer[] layers;
	
	private double error;
	
	public static FeedforwardNetwork createNetwork(double learningRate, int inputNodes, int hidden1, int outputNodes) {
		FeedforwardNetwork network = new FeedforwardNetwork(learningRate, inputNodes, hidden1, outputNodes);
		
		return network;		
	}

	private FeedforwardNetwork(double learningRate, int... nodes) {
		this.learningRate = learningRate;
		this.layers = new FeedforwardActiveLayer [nodes.length - 1];
		
		inputLayer = new FeedforwardInputLayer(nodes [0]);
		
		for (int i = 1; i < nodes.length; i++) {
			if (i == nodes.length - 1) {
				layers [i - 1] = new FeedforwardActiveLayer(FeedforwardActiveLayer.Type.OUTPUT, nodes [i], nodes [i - 1]);
			}
			else {
				layers [i - 1] = new FeedforwardActiveLayer(FeedforwardActiveLayer.Type.HIDDEN, nodes [i], nodes [i - 1]);
			}
		}
	}
	
	public double[] calculateOutputs(double[] input) {
		inputLayer.setOutputs(input);
		
		layers [0].calculateOutputs(inputLayer.getOutputs());
		for (int i = 1; i < layers.length; i++) {
			layers [i].calculateOutputs(layers [i - 1].getOutputs());
		}
		
		return layers [layers.length - 1].getOutputs();
	}
	
	public double trainBatch(double[][] input, double[][] targets) {
		for (int i = 0; i < layers.length; i++) {
			layers [i].resetAccumulatedErrorDelta();
		}
		
		for (int i = 0; i < input.length; i++) {
			calculateOutputs(input [i]);
			backpropagationNodeDeltas(targets [i]);

			layers [0].incrementErrorDelta(inputLayer);
			for (int j = 1; j < layers.length; j++) {
				layers [j].incrementErrorDelta(layers [j - 1]);
			}			
		}
		
		for (int j = 0; j < layers.length; j++) {
			layers [j].updateWeightsBatch(learningRate);
		}

		ErrorCalculation errorCalculation = new ErrorCalculation();
		
		for (int i = 0; i < input.length; i++) {
			double[] output = calculateOutputs(input [i]);
			errorCalculation.updateError(output, targets [i]);
		}

		error = errorCalculation.calculateError();
		
		return error;
	}
	
	private void backpropagationNodeDeltas(double[] target) {
		// output layer
		layers [layers.length - 1].backpropagationNodeDeltas(target);
		
		for (int i = layers.length - 2; i >= 0; i--) {
			layers [i].backpropagationNodeDeltas(layers [i + 1]);
		}
	}

	@Override
	public String toString() {
		String s = "";
		
		for (int i = 0; i < layers.length; i++) {
			s += layers [i];
			
			if (i != layers.length - 1) {
				s += "\n";
			}
		}
		
		return s;
	}
	
}
