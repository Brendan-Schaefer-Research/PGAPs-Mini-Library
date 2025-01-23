function ONELAY(ARR,weightsarr,biasesarr) {

	let returnarr = matrixmult([ARR],weightsarr)[0];
	if (addbias == true && biasesarr !== undefined) {
		returnarr = add2d([returnarr],[biasesarr])[0];
	}

	return returnarr;

}//takes in 1d array and returns one transform with weights from layer + bias
