package neuralnetwork.activation;

public class ActivationSigmoid {

	private static final ActivationSigmoid instance = new ActivationSigmoid();
	
	private ActivationSigmoid() {
	}
	
	public static ActivationSigmoid getInstance() {
		return instance;
	}
	
	public double activationFunction(double a) {
		return 1 / (1 + Math.exp(-1 * a));
	}

	public double derivativeFunction(double y) {
		return y * (1 - y);
	}
	
}
