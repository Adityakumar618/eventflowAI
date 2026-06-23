import { ChevronDown } from 'lucide-react'

export default function FormSelect({ value, onChange, children, className = '', ...props }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={onChange}
        className={`storm-input storm-select w-full appearance-none rounded-lg py-2 pl-3 pr-9 ${className}`}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        size={16}
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 opacity-50"
        style={{ color: 'var(--text-muted)' }}
      />
    </div>
  )
}