package neuralnetwork.layer;

public class FeedforwardInputLayer implements FeedforwardLayer {

	private double[] outputs;
	
	public FeedforwardInputLayer(int numNodes) {
		this.outputs = new double [numNodes + 1];
		this.outputs [numNodes] = 1;
	}
	
	public void setOutputs(double[] input) {
		for (int i = 0; i < input.length; i++) {
			outputs [i] = input [i];
		}
	}
	
	@Override
	public int getNumNodes() {
		return outputs.length;
	}

	@Override
	public double[] getOutputs() {
		return outputs;
	}
	
}
