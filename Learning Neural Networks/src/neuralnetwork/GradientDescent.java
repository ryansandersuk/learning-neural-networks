package neuralnetwork;

import java.text.NumberFormat;

public class GradientDescent {

	private static final NumberFormat NUMBER_FORMAT = NumberFormat.getInstance();
	
	public static void main(String[] args) {
		NUMBER_FORMAT.setMinimumFractionDigits(6);
		
		double xOld = 0;
		double xNew = 6;
		double proportion = 0.1;
		double precision = 0.00001;
		
		while (Math.abs(xNew - xOld) > precision) {
			xOld = xNew;
			double change = fPrime(xOld);
			
			xNew = xOld - proportion * change;
			
			System.out.println("Change: " + NUMBER_FORMAT.format(change) +
				", xNew = " + NUMBER_FORMAT.format(xNew));
		}
		
		System.out.println("Local minimum occurs at: " + NUMBER_FORMAT.format(xNew));
	}
	
	private static double fPrime(double x) {
		return 2 * x;
	}

}
