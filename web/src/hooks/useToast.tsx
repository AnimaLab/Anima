import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react'
import { Check, X } from 'lucide-react'

interface Toast {
  msg: string
  error?: boolean
}

interface ToastContextValue {
  show: (msg: string, error?: boolean) => void
}

const ToastContext = createContext<ToastContextValue>({ show: () => {} })

export function useToast() {
  return useContext(ToastContext)
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<Toast | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(null)

  const show = useCallback((msg: string, error = false) => {
    if (timer.current) clearTimeout(timer.current)
    setToast({ msg, error })
    timer.current = setTimeout(() => setToast(null), 4000)
  }, [])

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 animate-in fade-in slide-in-from-bottom-2">
          <div className={`flex items-center gap-2 text-white text-xs font-medium px-4 py-2.5 rounded-xl shadow-lg max-w-md ${toast.error ? 'bg-red-600' : 'bg-ink'}`}>
            {toast.error ? <X className="w-3.5 h-3.5 shrink-0" /> : <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />}
            <span className="truncate">{toast.msg}</span>
          </div>
        </div>
      )}
    </ToastContext.Provider>
  )
}
