import { Layers } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TOP_K_OPTIONS, type TopK } from "@/lib/top-k";

type Props = {
  value: TopK;
  onChange: (k: TopK) => void;
  disabled?: boolean;
  compact?: boolean;
};

export function TopKSelector({ value, onChange, disabled, compact }: Props) {
  return (
    <div className={compact ? "flex items-center gap-2" : "space-y-2"}>
      {!compact && (
        <label className="text-xs font-mono text-muted-foreground tracking-wider uppercase flex items-center gap-1.5">
          <Layers className="h-3.5 w-3.5" /> Top-K retrieval
        </label>
      )}
      <Select
        value={String(value)}
        onValueChange={(v) => onChange(Number(v) as TopK)}
        disabled={disabled}
      >
        <SelectTrigger
          className={
            compact
              ? "h-8 w-[72px] border-border bg-input/80 text-xs font-mono"
              : "border-border bg-input/80 font-mono"
          }
        >
          <SelectValue placeholder="K" />
        </SelectTrigger>
        <SelectContent className="border-border bg-popover/95 backdrop-blur-md">
          {TOP_K_OPTIONS.map((k) => (
            <SelectItem key={k} value={String(k)} className="font-mono text-sm">
              K = {k}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {!compact && (
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          Number of nearest neighbors returned by FAISS. Changing K re-runs retrieval and updates the embedding map.
        </p>
      )}
    </div>
  );
}
