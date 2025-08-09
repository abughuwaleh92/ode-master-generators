import React from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

export default function LatexRenderer({ latex, display = false }) {
  const html = React.useMemo(() => {
    try {
      // Convert SymPy output to LaTeX if needed
      let processedLatex = latex
      
      // Replace common SymPy patterns
      processedLatex = processedLatex
        .replace(/Eq\((.*?),\s*(.*?)\)/g, '$1 = $2')
        .replace(/Derivative\((.*?),\s*x,\s*(\d+)\)/g, '\\frac{d^{$2}$1}{dx^{$2}}')
        .replace(/Derivative\((.*?),\s*x\)/g, '\\frac{d$1}{dx}')
        .replace(/y\(x\)/g, 'y')
        .replace(/\*\*/g, '^')
        
      return katex.renderToString(processedLatex, {
        displayMode: display,
        throwOnError: false,
      })
    } catch (error) {
      console.error('LaTeX render error:', error)
      return `<span class="text-red-500">Error rendering: ${latex}</span>`
    }
  }, [latex, display])

  return (
    <div
      className={`${display ? 'text-center my-4' : 'inline-block'}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
