import React, { useState } from 'react'
import { useMutation } from 'react-query'
import { api } from '../../api/client'
import { CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import Editor from 'react-simple-code-editor'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import 'prismjs/themes/prism.css'

export default function VerificationPanel() {
  const [ode, setOde] = useState('Eq(Derivative(y(x), x, 2) + y(x), sin(x))')
  const [solution, setSolution] = useState('sin(x)')
  const [method, setMethod] = useState('substitution')
  const [result, setResult] = useState(null)

  const verifyMutation = useMutation(api.verify, {
    onSuccess: (data) => {
      setResult(data)
      if (data.verified) {
        toast.success('ODE verified successfully!')
      } else {
        toast.error('Verification failed')
      }
    },
  })

  const handleVerify = () => {
    if (!ode || !solution) {
      toast.error('Please provide both ODE and solution')
      return
    }
    verifyMutation.mutate({ ode, solution, method })
  }

  const examples = [
    {
      name: 'Linear ODE',
      ode: "Eq(Derivative(y(x), x, 2) + y(x), sin(x))",
      solution: "sin(x)/2 - x*cos(x)/2"
    },
    {
      name: 'Nonlinear ODE',
      ode: "Eq(Derivative(y(x), x)**2 + y(x), exp(x))",
      solution: "exp(x) - 1"
    },
    {
      name: 'First Order',
      ode: "Eq(Derivative(y(x), x) + 2*y(x), exp(x))",
      solution: "exp(x)/3"
    }
  ]

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-6">ODE Verification</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                ODE Equation (SymPy format)
              </label>
              <div className="border rounded-lg overflow-hidden">
                <Editor
                  value={ode}
                  onValueChange={setOde}
                  highlight={code => Prism.highlight(code, Prism.languages.python, 'python')}
                  padding={10}
                  style={{
                    fontFamily: '"Fira code", "Fira Mono", monospace',
                    fontSize: 14,
                    minHeight: 100,
                  }}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Solution Expression
              </label>
              <div className="border rounded-lg overflow-hidden">
                <Editor
                  value={solution}
                  onValueChange={setSolution}
                  highlight={code => Prism.highlight(code, Prism.languages.python, 'python')}
                  padding={10}
                  style={{
                    fontFamily: '"Fira code", "Fira Mono", monospace',
                    fontSize: 14,
                    minHeight: 60,
                  }}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Verification Method
              </label>
              <select
                className="select"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
              >
                <option value="substitution">Substitution</option>
                <option value="numeric">Numerical</option>
                <option value="symbolic">Symbolic</option>
              </select>
            </div>

            <button
              onClick={handleVerify}
              disabled={verifyMutation.isLoading}
              className="w-full btn-primary"
            >
              {verifyMutation.isLoading ? 'Verifying...' : 'Verify ODE'}
            </button>
          </div>

          {/* Examples Section */}
          <div>
            <h3 className="font-semibold mb-3">Examples</h3>
            <div className="space-y-3">
              {examples.map((example, idx) => (
                <div
                  key={idx}
                  className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    setOde(example.ode)
                    setSolution(example.solution)
                  }}
                >
                  <h4 className="font-medium text-sm mb-1">{example.name}</h4>
                  <div className="text-xs text-gray-600 font-mono">
                    <div className="truncate">ODE: {example.ode}</div>
                    <div className="truncate">Sol: {example.solution}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Help Section */}
            <div className="mt-6 bg-blue-50 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2">
                <AlertCircle className="w-4 h-4 inline mr-1" />
                SymPy Format Guide
              </h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Use Eq() for equations</li>
                <li>• Derivative(y(x), x) for dy/dx</li>
                <li>• Derivative(y(x), x, 2) for d²y/dx²</li>
                <li>• Standard functions: sin, cos, exp, log</li>
                <li>• Operations: +, -, *, /, **</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Result Section */}
      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Verification Result</h3>
          
          <div className={`rounded-lg p-4 ${result.verified ? 'bg-green-50' : 'bg-red-50'}`}>
            <div className="flex items-start">
              {result.verified ? (
                <CheckCircle className="w-6 h-6 text-green-600 mt-0.5 mr-3" />
              ) : (
                <XCircle className="w-6 h-6 text-red-600 mt-0.5 mr-3" />
              )}
              
              <div className="flex-1">
                <h4 className={`font-medium text-lg ${result.verified ? 'text-green-900' : 'text-red-900'}`}>
                  {result.verified ? 'Verification Successful' : 'Verification Failed'}
                </h4>
                
                <div className="mt-2 space-y-2 text-sm">
                  <div>
                    <span className="font-medium">Method:</span> {result.method}
                  </div>
                  <div>
                    <span className="font-medium">Confidence:</span> {(result.confidence * 100).toFixed(1)}%
                  </div>
                  {result.error && (
                    <div>
                      <span className="font-medium">Error:</span> {result.error}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
