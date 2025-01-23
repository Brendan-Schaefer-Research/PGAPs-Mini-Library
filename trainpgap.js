function backprop(allweights,allbiases,nstore,costpertoken,activated) {
	
	for (bb = allweights.length-1; bb >= 0; bb--) {//layer
		for (aa = 0; aa < allweights[bb].length; aa++) {//first neuron
			for (aa1 = 0; aa1 < allweights[bb][aa].length; aa1++) {//second neuron
				let gfd = getfuncderiv(nstore[bb+1][aa1]);
				if (activated == false) {
					gfd = 1;
				}
				allweights[bb][aa][aa1] += 
					activate([nstore[bb][aa]])[0] * //in terms of zl- prev neuron is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[bb+1][aa1] *  //in terms of cost- desired change to cost
					learningrate;
				allbiases[bb][aa1] += 
					gfd * //in terms of al- derivative of prev w/ respect to zl
					costpertoken[bb+1][aa1] *  //in terms of cost- desired change to cost down the line
					learningrate;
				costpertoken[bb][aa] += 
					allweights[bb][aa][aa1] * //in terms of zl- weight is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[bb+1][aa1];  //in terms of cost- desired change to cost down the line
			}
		}
	}
	return costpertoken;
	
}

function trainPGAP(disp) {
	
	let costarr = runexample(getinput(),false);
	
	//encoders
	let decoders = transpose(CA(encoders));
	let incost = [maketensor(1,[encodesize],0),CA(costarr)];
	backprop([decoders],[maketensor(1,[tokens.length],0)],encneur,incost);
	encoders = transpose(decoders);
	costarr = maketensor(1,[iterations],normalize(CA(incost[0]),sureness));
	
	for (ll = layers-1; ll >= 0; ll--) {
		//kickbacks
		let incost = [maketensor(1,[iterations*encodesize],0),concatenate(CA(costarr))];
		let outcost = backprop([apweights[ll]],[apbiases[ll]],apneur[ll],incost)[0];
		for (ii = 0; ii < iterations; ii++) {
			let attencosts = normalize(outcost.slice(ii*encodesize,(ii+1)*encodesize),sureness);
			costarr[ii] = concatenate([attencosts,CA(costarr[ii])])
		}//extend costarr
		
		//attention network backprop
		for (ii = 0; ii < iterations; ii++) {
			//set up costin
			let costin = maketensor(2,[ffnlayers+2,2*encodesize],0);//hidden and last layer
			costin[0] = maketensor(1,[encodesize],0);//first layer
			costin[ffnlayers+1] = CA(costarr[ii]);//last layer
			
			//backpropagate
			backprop(aweights[ll][ii],abiases[ll][ii],neuronstore[ll][ii],costin)
			costarr[ii] = normalize(CA(costin[0]),sureness);
		}
	}//attennet
	opxd("add",identities,opxd("mult",CA(costarr),maketensor(2,[iterations,encodesize],learningrate)));//iterations
	
	runexample(getinput(),disp == 0);//for display
	
}
