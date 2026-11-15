import type { AskRequest, ResultEvent } from './api'

export interface HistoryEntry {
  id: string
  ts: number
  request: AskRequest
  result: ResultEvent
}

const KEY = 'mug.history.v1'
const MAX = 25

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
  } catch {
    return []
  }
}

export function saveHistory(entries: HistoryEntry[]): HistoryEntry[] {
  let list = entries.slice(0, MAX)
  // Shrink until it fits the storage quota.
  while (list.length) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list))
      return list
    } catch {
      list = list.slice(0, list.length - 1)
    }
  }
  localStorage.removeItem(KEY)
  return list
}
