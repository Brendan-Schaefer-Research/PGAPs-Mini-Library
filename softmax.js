function SOFTMAX(ARR) {
	
	let exsum = 0;
	let arrtoreturn = [];
	for (g = 0; g < ARR.length; g++) {
		exsum += pow(e,ARR[g]/smtemperature);
	}
	for (g = 0; g < ARR.length; g++) {
		arrtoreturn[g] = pow(e,ARR[g]/smtemperature)/exsum;
		if (isNaN(arrtoreturn[g])) {
			arrtoreturn[g] = 1;
		}
	}
	return arrtoreturn;
	
}
