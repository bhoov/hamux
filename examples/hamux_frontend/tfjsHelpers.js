// Bypass finite-ness checking
const IsFinite = (node) => {
    if (node.inputs) {
    return tf.logicalNot(tf.logicalOr(tf.isNaN(node.inputs[0]), tf.isInf(node.inputs[0])))
    }
    return true
    // return true
}

tf.registerOp('IsFinite', IsFinite);

export function inputSortOrder(inpNames) {
    return (inpVals) => {
        const catNames = R.zip(R.range(0, inpNames.length), inpNames)
        const orderedIdxs = R.sortBy(a => a[1])(catNames).map(x => x[0])

        let modelInput = R.range(0, inpNames.length)
        R.zip(orderedIdxs, inpVals).forEach(([i, val]) => {
            modelInput[i] = val
        })
        return modelInput
    }
}

/**
 * Order the output according to the expected output ordering
*/
export function outputSortOrder(outputNames, expOutputOrder) {
    const catNames = R.zip(R.range(0, outputNames.length), outputNames)
    console.log("output", catNames)
    const orderedIdxs = R.sortBy(a => a[1])(catNames).map(x => x[0])
    console.log(orderedIdxs)

    return (outputs) => {
        let obj = {}
        orderedIdxs.forEach((x, i) => {
            obj[expOutputOrder[i]] = outputs[x]
        })
        return obj
    }
}

export function TFJSInterface(model, expectedOutputs) {
    const parseInput = inputSortOrder(Object.keys(model.artifacts.signature.inputs))
    const parseOutput = outputSortOrder(Object.values(model.artifacts.signature.outputs).map(o => o.name), expectedOutputs)
    console.log("inputs: ", parseInput(["img", "logits", "dt"]))
    function predict(inputs) {
        return parseOutput(model.predict(parseInput(inputs)))
    }
    return {
        parseInput,
        parseOutput,
        predict
    }
}