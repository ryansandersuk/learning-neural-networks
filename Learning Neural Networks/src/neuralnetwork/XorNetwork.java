package neuralnetwork;

import java.text.NumberFormat;

public class XorNetwork {

	private static double[][] XOR_INPUT = 
		{{0, 0},
		 {1, 0},
		 {0, 1},
		 {1, 1}};
	private static double[][] XOR_IDEAL = 
		{{0},
		 {1},
		 {1},
		 {0}};
	
	public static void main(String[] args) {
		FeedforwardNetwork network = FeedforwardNetwork.createNetwork(0.7, 2, 2, 1);
		int epoch = 0;

		while (epoch < 5000) {
			double error = network.trainBatch(XOR_INPUT, XOR_IDEAL);
			
			System.out.println("Epoch #" + epoch + ": " + error);
			epoch++;
		}

		System.out.println("Network:\n" + network);

		System.out.println("Testing results...");
		NumberFormat format = NumberFormat.getNumberInstance();
		format.setMinimumFractionDigits(4);

		for (int i = 0; i < XOR_IDEAL.length; i++) {
			double[] predicted = network.calculateOutputs(XOR_INPUT [i]);
			
			for (int j = 0; j < XOR_IDEAL [0].length; j++) {
				double error = XOR_IDEAL [i][j] - predicted [j];
				System.out.println(i + ", " + j +
						": predicted=" + format.format(predicted [j]) +
						", XOR_IDEAL=" + format.format(XOR_IDEAL [i][j]) +
						", error=" + format.format(error));						
			}
		}
	}

}
