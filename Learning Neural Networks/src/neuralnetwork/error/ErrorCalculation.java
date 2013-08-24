package neuralnetwork.error;

public class ErrorCalculation {

	private double globalError;
	
	public ErrorCalculation() {
		globalError = 0.0;
	}
	
	public void updateError(double[] actual, double[] ideal) {
		for (int i = 0; i < ideal.length; i++) {
			double delta = ideal [i] - actual [i];
			globalError += Math.pow(delta, 2);			
		}
	}
	
	public double calculateError() {
		return globalError;
	}
	
}
