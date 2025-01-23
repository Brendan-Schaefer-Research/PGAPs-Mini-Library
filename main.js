function LINEAR(input,allwei,allbi) {

  return runlinear(input,allwei.length,false,[allwei],[allbi]);

}

function runPGAP(input) {

	let last = [];
	for (ii = 0; ii < learningset; ii++) {
		last[ii] = CA(encoders[input[ii]]);
	}//vector encoding
	
	for (ii = 0; ii < iterations; ii++) {
		opxd("add",last[ii],identities[ii]);
		opxd("add",last[ii],maketensor(1,[encodesize],positioners(ii+1)));
	}//identities and pos encoding
	
	for (ll = 0; ll < layers; ll++) {
		//run attention network
		let kprod = [];
		for (ii = 0; ii < iterations; ii++) {
			//get attention and result
			let linout = runlinear(CA(last[ii]),aweights[ll][ii].length,false,aweights[ll][ii],abiases[ll][ii]);//get linear output
			opxd("sub",linout[0],maketensor(1,[linout[0].length],1/2));//scale entire last layer to avg 0
			last[ii] = CA(linout[0]).slice(0,encodesize);//add first part to last
			
			//store values
			neuronstore[ll][ii] = linout[1];
			kprod[ii] = CA(linout[0]).slice(encodesize,2*encodesize);
		}
		
		//run connector network
		let attenout = runlinear(concatenate(kprod),1,false,[apweights[ll]],[apbiases[ll]]);
		let sumkprod = opxd("sub",attenout[0],maketensor(1,[attenout[0].length],1/2));//scale entire last layer to avg 0
		apneur[ll] = attenout[1];

		for (ii = 0; ii < iterations; ii++) {
			opxd("add",last[ii],sumkprod.slice(ii*encodesize,(ii+1)*encodesize));
		}//add corresponding section of attention to last
		
	}//pgap
	
	//calculate final probabilities
	let avglast = maketensor(1,[encodesize],0);
	for (a = 0; a < learningset; a++) {
		opxd("add",avglast,last[a]);
	}//sum
	opxd("div",avglast,maketensor(1,[encodesize],learningset));//divide by learningset
	let encout = runlinear(CA(avglast),1,false,[transpose(CA(encoders))],[]);
	encneur = encout[1];
	
	return encout[0];
	
}


function SHAPE(ARR1) {

  let shapeARR = [ARR1.length];
	for (ga = 0; ga < 6; ga++) {
    let testv = dimen(false,ARR1,maketensor(1,[ga+2],0));
		if (testv === undefined) {
			break;
		}
	  shapeARR[ga+1] = testv.length;
	}
  return shapeARR;

}

function TENSOR(fill,shapearr) {

  return maketensor(shapearr.length,shapearr,fill);

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
