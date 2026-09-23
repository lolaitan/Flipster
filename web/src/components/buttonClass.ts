import clsx from 'clsx'

export type Variant = 'primary' | 'secondary' | 'ghost'

export function buttonClass(variant: Variant = 'secondary', size: 'sm' | 'md' = 'md', className?: string) {
  return clsx(
    'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
    size === 'sm' ? 'h-8 px-2.5 text-sm' : 'h-10 px-4 text-sm',
    variant === 'primary' && 'bg-accent text-white shadow-sm hover:bg-accent-dark',
    variant === 'secondary' && 'border border-line bg-paper text-ink hover:bg-paper-2',
    variant === 'ghost' && 'text-ink-2 hover:bg-paper-2 hover:text-ink',
    className,
  )
}
