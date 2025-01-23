function onelay(ARR,weightsarr,biasesarr) {

	let returnarr = matrixmult([ARR],weightsarr)[0];
	if (addbias == true && biasesarr !== undefined) {
		returnarr = add2d([returnarr],[biasesarr])[0];
	}

	return returnarr;

}//takes in 1d array and returns one transform with weights from layer + bias

function LINEAR(input,qlayers,sorted,allweights,allbiases) {

	let nsra = [];
	nsra[0] = input;
	let ra = input;
	for (a = 0; a < qlayers; a++) {
		nsra[a+1] = onelay(CA(ra),allweights[a],allbiases[a]);
		ra = activate(nsra[a+1]);
	}
	if (sorted == true) {
		return [Bsort(CA(ra),outputarr),nsra];
	}
	else {
		return [CA(ra),nsra];
	}

}//takes in parameters for layers (corr to weights), if sorted, input 
//returns an array of [output , each layer unactivated arr]
