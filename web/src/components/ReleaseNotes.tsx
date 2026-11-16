import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { X, ArrowUpCircle, ExternalLink } from 'lucide-react'
import type { UpdateStatus } from '../api/types'

/** Release notes of the latest release, as published on GitHub. */
export function ReleaseNotes({ update, onClose }: { update: UpdateStatus; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const published = update.published_at ? new Date(update.published_at).toLocaleDateString() : null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="release-notes-title"
        className="bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 px-6 pt-5 pb-3 border-b border-warm-border">
          <div className="flex items-start gap-3 min-w-0">
            <ArrowUpCircle size={20} className="shrink-0 text-accent mt-0.5" />
            <div className="min-w-0">
              <h2 id="release-notes-title" className="text-lg font-semibold text-ink">
                {update.name || `Anima v${update.latest}`}
              </h2>
              <p className="text-[13px] text-ink-muted mt-0.5">
                {update.update_available ? `You run v${update.current}.` : `You run the latest version.`}
                {published && ` Published ${published}.`}
              </p>
            </div>
          </div>
          <button ref={closeRef} onClick={onClose} aria-label="Close" className="btn btn-ghost btn-icon -mr-2">
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-5 overflow-y-auto text-sm text-ink-light leading-relaxed">
          {update.notes ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-5 mb-3 last:mb-0 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 last:mb-0 space-y-1">{children}</ol>,
                h1: ({ children }) => <h3 className="text-base font-semibold text-ink mt-5 first:mt-0 mb-2">{children}</h3>,
                h2: ({ children }) => <h3 className="text-base font-semibold text-ink mt-5 first:mt-0 mb-2">{children}</h3>,
                h3: ({ children }) => <h4 className="text-[15px] font-semibold text-ink mt-4 first:mt-0 mb-1">{children}</h4>,
                strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
                code: ({ children }) => <code className="bg-paper-deep text-ink px-1.5 py-0.5 rounded text-[13px] break-words">{children}</code>,
                pre: ({ children }) => <pre className="bg-paper-deep rounded-lg p-3 my-3 overflow-x-auto text-[13px]">{children}</pre>,
                a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-2">{children}</a>,
                blockquote: ({ children }) => <blockquote className="border-l-2 border-warm-border-strong pl-3 text-ink-muted my-3">{children}</blockquote>,
                hr: () => <hr className="border-warm-border my-4" />,
                table: ({ children }) => <div className="my-3 overflow-x-auto"><table className="border-collapse text-[13px] w-full">{children}</table></div>,
                th: ({ children }) => <th className="border border-warm-border px-2.5 py-1.5 bg-paper-deep text-left font-medium">{children}</th>,
                td: ({ children }) => <td className="border border-warm-border px-2.5 py-1.5">{children}</td>,
              }}
            >
              {update.notes}
            </ReactMarkdown>
          ) : (
            <p className="text-ink-muted">This release has no notes. The release page may have more.</p>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 px-6 py-3 border-t border-warm-border bg-paper-warm">
          {update.url && (
            <a href={update.url} target="_blank" rel="noopener noreferrer" className="btn btn-secondary">
              <ExternalLink size={14} /> View on GitHub
            </a>
          )}
          <button onClick={onClose} className="btn btn-primary">Close</button>
        </div>
      </div>
    </div>
  )
}
