import { getPath, setPath, type PathPart } from "./configForm";

export type ObjectFieldKind = "boolean" | "number" | "shortText" | "longText" | "simpleArray";
export type PrimitiveArrayMode = "expand" | "field";

export interface ObjectFormField {
  path: PathPart[];
  label: string;
  kind: ObjectFieldKind;
  value: unknown;
  help: string;
}

export interface ObjectFormGroup {
  key: string;
  title: string;
  path: PathPart[];
  fields: ObjectFormField[];
}

export interface ObjectArrayGroup {
  key: string;
  title: string;
  path: PathPart[];
  items: unknown[];
  template: unknown;
}

export interface CollectObjectFieldsOptions {
  excludeRootKeys?: string[];
  helpForField?: (path: PathPart[]) => string;
  primitiveArrayMode?: PrimitiveArrayMode;
}

export function collectObjectFields(value: unknown, options: CollectObjectFieldsOptions = {}, path: PathPart[] = []): ObjectFormField[] {
  if (typeof value === "boolean") return [field(path, "boolean", value, options)];
  if (typeof value === "number") return [field(path, "number", value, options)];
  if (typeof value === "string") return [field(path, value.length > 90 || value.includes("\n") ? "longText" : "shortText", value, options)];
  if (Array.isArray(value)) {
    if (options.primitiveArrayMode === "field" && value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
      return [field(path, "simpleArray", value, options)];
    }
    return value.flatMap((item, index) => collectObjectFields(item, options, [...path, index]));
  }
  if (value && typeof value === "object") {
    const excluded = new Set(options.excludeRootKeys ?? []);
    return Object.entries(value as Record<string, unknown>)
      .filter(([key]) => path.length > 0 || !excluded.has(key))
      .flatMap(([key, child]) => collectObjectFields(child, options, [...path, key]));
  }
  return [];
}

export function groupObjectFields(fields: ObjectFormField[]): ObjectFormGroup[] {
  const groups = new Map<string, ObjectFormGroup>();
  for (const field of fields) {
    const groupPath = field.path.length > 1 ? field.path.slice(0, -1) : [];
    const key = groupPath.join(".") || "general";
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        path: groupPath,
        title: groupPath.length ? labelForPath(groupPath) : "General",
        fields: []
      });
    }
    groups.get(key)?.fields.push(field);
  }
  return [...groups.values()];
}

export function collectArrayGroups(value: unknown, templates: Record<string, unknown>): ObjectArrayGroup[] {
  if (!value || typeof value !== "object") return [];
  return Object.entries(templates).map(([key, template]) => ({
    key,
    title: labelForPath([key]),
    path: [key],
    items: arrayPathItems(value, [key]),
    template
  }));
}

export function updateObjectField(target: Record<string, unknown>, field: ObjectFormField, rawValue: string | number | boolean) {
  let value: unknown = rawValue;
  if (field.kind === "number") value = Number(rawValue);
  if (field.kind === "simpleArray") {
    value = String(rawValue)
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  setPath(target, field.path, value);
}

export function displayObjectFieldValue(field: ObjectFormField): string {
  return Array.isArray(field.value) ? field.value.join(", ") : String(field.value ?? "");
}

export function fieldLabel(field: ObjectFormField, group: ObjectFormGroup): string {
  return labelForPath(field.path.slice(group.path.length));
}

export function labelForPath(path: PathPart[]): string {
  return path.map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " "))).join(" / ");
}

function field(path: PathPart[], kind: ObjectFieldKind, value: unknown, options: CollectObjectFieldsOptions): ObjectFormField {
  return { path, label: labelForPath(path), kind, value, help: options.helpForField?.(path) ?? "" };
}

function arrayPathItems(target: unknown, path: PathPart[]): unknown[] {
  const value = getPath(target, path);
  return Array.isArray(value) ? value : [];
}
