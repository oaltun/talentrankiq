import { useMemo } from "react"
import type { ChunkSim } from "../../../../types"

interface HighlightedCvProps {
  text: string
  chunks: ChunkSim[]
  model: string
}

export function HighlightedCv({ text, chunks, model }: HighlightedCvProps) {
  const highlightedText = useMemo(() => {
    if (!text)
      return <span className="text-muted-foreground">No text extracted.</span>

    // Sort chunks by char_start
    const sortedChunks = [...chunks].sort((a, b) => a.char_start - b.char_start)

    const parts: React.ReactNode[] = []
    let lastIndex = 0

    for (const chunk of sortedChunks) {
      // Add text before this chunk
      if (chunk.char_start > lastIndex) {
        parts.push(text.slice(lastIndex, chunk.char_start))
      }

      // Add highlighted chunk
      const chunkText = text.slice(chunk.char_start, chunk.char_end)
      const intensity = Math.min(chunk.similarity * 1.5, 1)
      const bgColor = `rgba(59, 130, 246, ${0.1 + intensity * 0.3})`
      const borderColor = `rgba(59, 130, 246, ${0.3 + intensity * 0.5})`

      parts.push(
        <span
          key={chunk.char_start}
          style={{
            backgroundColor: bgColor,
            borderBottom: `2px solid ${borderColor}`,
            padding: "0 2px",
            borderRadius: "2px",
          }}
          title={`Similarity: ${(chunk.similarity * 100).toFixed(1)}%`}
        >
          {chunkText}
        </span>
      )

      lastIndex = chunk.char_end
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex))
    }

    return (
      <span className="font-mono text-sm whitespace-pre-wrap">{parts}</span>
    )
  }, [text, chunks])

  return (
    <div className="max-h-[60vh] overflow-auto rounded border bg-background p-4">
      <p className="mb-2 text-xs text-muted-foreground">Model: {model}</p>
      {highlightedText}
    </div>
  )
}
