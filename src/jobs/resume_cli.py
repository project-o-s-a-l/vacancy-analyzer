"""CLI uses the same preparation, review, revision and confirmation service as Telegram."""

import asyncio

from bot.resume_cards import render_resume, render_resume_details
from jobs.resume_analysis import resume_analyzer
from jobs.resumes import ResumeService
from processing.resume import MAX_RESUME_BYTES, read_resume
from storage.repositories.user_repository import UserRepository


async def run_resume_cli(args, settings, sessions):
    async with sessions() as session:
        user = await UserRepository(session).get(args.user_id)
    if user is None:
        raise ValueError("Сначала зарегистрируйтесь в боте командой /start.")
    async with resume_analyzer(settings) as analyzer:
        service = ResumeService(
            sessions,
            analyzer=analyzer,
            queued=True,
            model=settings.ollama_model if settings.resume_llm_enabled else "rules",
        )
        if args.delete:
            await service.delete(args.user_id)
            print("Текст резюме и черновик удалены; профиль сохранён.")
            return
        if args.cancel:
            await service.cancel(args.user_id)
            print("Загрузка отменена; профиль сохранён.")
            return
        if args.file:
            with args.file.open("rb") as stream:
                text, kind = read_resume(stream.read(MAX_RESUME_BYTES + 1), args.file.name)
            await service.submit(args.user_id, text, format=kind)
            print("Резюме сохранено в очереди. Запустите worker; затем откройте resume без --file.")
            return
        else:
            draft = await service.pending(args.user_id)
        if draft is None:
            status = await service.status(args.user_id)
            if status:
                import json

                print(json.dumps(status, ensure_ascii=False, indent=2))
                return
            raise ValueError("Нет черновика; загрузите резюме с --file.")
        if args.edit:
            draft = await service.revise(args.user_id, draft.token, *args.edit)
        print(render_resume(draft, user.profile))
        if args.details:
            print(render_resume_details(draft))
        if args.apply:
            answer = await asyncio.to_thread(input, "Применить показанный результат? Введите да: ")
            if answer.strip().casefold() == "да":
                await service.confirm(args.user_id, draft.token)
                print("Профиль обновлён. Следующий поиск использует новые сведения.")
            else:
                print("Черновик сохранён без применения.")
