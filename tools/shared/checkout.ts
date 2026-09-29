/**
 * 手元の checkout の版と未コミットの変更の有無（Deno 専用 — `git` を起動する）。
 *
 * GPU lab のページの `/config.json`（`tools/gpu-lab/server.ts`）と Deno の双子 CLI
 * （`tools/geometry-sweep/main.ts`・`tools/anima-residency/profile.ts`）の JSON が同じ取り方で載せる。
 * どの道具にも属さない場所に置くのは、道具どうしが互いの入口を import し合わないため。
 */
export const readCheckout = async (): Promise<{ revision: string; dirty: boolean }> => {
  const git = await new Deno.Command("git", { args: ["rev-parse", "HEAD"] }).output();
  if (!git.success) throw Error("Cannot identify checkout revision");
  const status = await new Deno.Command("git", { args: ["status", "--porcelain"] }).output();
  if (!status.success) throw Error("Cannot inspect checkout changes");
  return {
    revision: new TextDecoder().decode(git.stdout).trim(),
    dirty: status.stdout.length > 0,
  };
};
