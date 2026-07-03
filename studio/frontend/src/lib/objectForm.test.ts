import { describe, expect, test } from "bun:test";
import {
  collectArrayGroups,
  collectObjectFields,
  displayObjectFieldValue,
  fieldLabel,
  groupObjectFields,
  labelForPath,
  updateObjectField
} from "./objectForm";

describe("object form helpers", () => {
  test("collects scalar fields and skips configured root keys", () => {
    const fields = collectObjectFields(
      {
        name: "Demo",
        enabled: true,
        count: 3,
        skipped: "no",
        nested: { long_text: "line one\nline two" }
      },
      {
        excludeRootKeys: ["skipped"],
        helpForField: (path) => `help:${path.join(".")}`
      }
    );

    expect(fields.map((field) => ({ path: field.path.join("."), kind: field.kind, help: field.help }))).toEqual([
      { path: "name", kind: "shortText", help: "help:name" },
      { path: "enabled", kind: "boolean", help: "help:enabled" },
      { path: "count", kind: "number", help: "help:count" },
      { path: "nested.long_text", kind: "longText", help: "help:nested.long_text" }
    ]);
  });

  test("supports primitive arrays as simple fields or expanded fields", () => {
    expect(collectObjectFields({ tags: ["a", "b"] }, { primitiveArrayMode: "field" }).map((field) => field.kind)).toEqual(["simpleArray"]);
    expect(collectObjectFields({ tags: ["a", "b"] }, { primitiveArrayMode: "expand" }).map((field) => field.path.join("."))).toEqual([
      "tags.0",
      "tags.1"
    ]);
  });

  test("groups fields by parent path and formats labels", () => {
    const fields = collectObjectFields({ video: { fps: 24 }, title: "Demo" });
    const groups = groupObjectFields(fields);

    expect(groups.map((group) => ({ key: group.key, title: group.title, labels: group.fields.map((field) => fieldLabel(field, group)) }))).toEqual([
      { key: "video", title: "video", labels: ["fps"] },
      { key: "general", title: "General", labels: ["title"] }
    ]);
    expect(labelForPath(["actors", 0, "image_prompt"])).toBe("actors / #1 / image prompt");
  });

  test("updates fields with typed conversions", () => {
    const target: Record<string, unknown> = { enabled: false, count: 1, tags: [] };
    const fields = collectObjectFields(target, { primitiveArrayMode: "field" });

    updateObjectField(target, fields.find((field) => field.path.join(".") === "enabled")!, true);
    updateObjectField(target, fields.find((field) => field.path.join(".") === "count")!, "5");
    updateObjectField(target, fields.find((field) => field.path.join(".") === "tags")!, "a, b, ");

    expect(target).toEqual({ enabled: true, count: 5, tags: ["a", "b"] });
  });

  test("collects array groups from templates and displays array values", () => {
    const groups = collectArrayGroups({ actors: [{ name: "Mara" }] }, { actors: { name: "" }, locations: { name: "" } });

    expect(groups.map((group) => ({ key: group.key, title: group.title, items: group.items }))).toEqual([
      { key: "actors", title: "actors", items: [{ name: "Mara" }] },
      { key: "locations", title: "locations", items: [] }
    ]);
    expect(displayObjectFieldValue({ path: ["tags"], label: "tags", kind: "simpleArray", value: ["a", "b"], help: "" })).toBe("a, b");
  });
});
