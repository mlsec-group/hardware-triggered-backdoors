namespace APIResponse {
    export interface QueueSummary {
        backends: { [key: string]: number };
        combinations: { [key: string]: number };
    }

    export interface Worker {
        id: string;
        state: "IDLE" | "BUSY" | "DEAD";
        client: Client;
        idle_time: number;
        busy_time: number;
        step_times: number[];
        wait_times: number[];
        between_step_times: number[];
        n_steps: number;
        n_assigned_jobs: number;
        current_capacity: number;
        max_jobs: number;
    }

    export type WorkerGroup = Worker[];

    export interface Job {
        id: string;
        iteration: number;
        n_steps: number;
        required_clients: string[];
    }

    export interface JobThread {
        worker_group: WorkerGroup;
        job: Job;
        runtime: number;
    }

    export interface Client {
        id: string;
    }

    export type Workers = { [key: string]: Worker };

    export interface JSON {
        progress: {
            iteration: number;
            total_steps: number | undefined;
        },
        n_clients_added: number;
        n_clients_removed: number;
        job_threads: JobThread[];
        queued_jobs: Job[];
        n_queued_jobs: number;
        redo_queued_jobs: Job[];
        n_redo_queued_jobs: number;
        redo_queue_summary: QueueSummary,
        queue_summary: QueueSummary,
        workers: { [key: string]: Worker };
        elapsed_time: number;
        max_queue: number;
    }
}

function avg(list: number[]) {
    if (list.length === 0) {
        return 0;
    }

    return list.reduce((a, b) => a + b) / list.length;
}

function aggregateTimes(workers: APIResponse.Workers) {
    const agg: {
        [key: string]: {
            busy: number;
            idle: number;
            wait: number;
            step: number;
            inter_step: number;
            n: number
        }
    } = {};
    for (const worker of Object.values(workers)) {
        if (agg[worker.client.id] === undefined) {
            agg[worker.client.id] = {
                idle: 0,
                busy: 0,
                wait: 0,
                step: 0,
                inter_step: 0,
                n: 0
            }
        }

        agg[worker.client.id].idle += worker.idle_time;
        agg[worker.client.id].busy += worker.busy_time;
        agg[worker.client.id].wait += avg(worker.wait_times);
        agg[worker.client.id].step += avg(worker.step_times);
        agg[worker.client.id].inter_step += avg(worker.between_step_times);

        agg[worker.client.id].n += 1;
    }
    return agg;
}

function updateBackendComponent(json: APIResponse.JSON, root: HTMLElement) {
    while (root.firstChild) {
        root.removeChild(root.firstChild);
    }

    const times = aggregateTimes(json.workers);
    const backends = Object.keys(times);

    const table = document.createElement("table");

    {
        const row = document.createElement("tr");
        const th0 = document.createElement("th");
        th0.textContent = "Backend";
        const th00 = document.createElement("th");
        th00.textContent = "Number of workers";
        const th1 = document.createElement("th");
        th1.textContent = "Idle";
        const th2 = document.createElement("th");
        th2.textContent = "Busy";
        const th3 = document.createElement("th");
        th3.textContent = "Wait";
        const th4 = document.createElement("th");
        th4.textContent = "Step";
        const th5 = document.createElement("th");
        th5.textContent = "Inter-Step";
        row.appendChild(th0);
        row.appendChild(th00);
        row.appendChild(th1);
        row.appendChild(th2);
        row.appendChild(th3);
        row.appendChild(th4);
        row.appendChild(th5);
        table.appendChild(row);
    }

    for (let i = 0; i < backends.length; i++) {
        const backend = backends[i];
        const row = document.createElement("tr");
        const cell0 = document.createElement("td");
        cell0.textContent = backend;

        const cell00 = document.createElement("td");
        cell00.textContent = String(Math.floor(times[backend].n));

        const cell1 = document.createElement("td");
        cell1.textContent = String(Math.floor(times[backend].idle));

        const cell2 = document.createElement("td");
        cell2.textContent = String(Math.floor(times[backend].busy));

        const cell3 = document.createElement("td");
        cell3.textContent = String(Math.floor(times[backend].wait * 1000)) + "ms";

        const cell4 = document.createElement("td");
        cell4.textContent = String(Math.floor(times[backend].step * 1000)) + "ms";

        const cell5 = document.createElement("td");
        cell5.textContent = String(Math.floor(times[backend].inter_step * 1000)) + "ms";

        row.appendChild(cell0);
        row.appendChild(cell00);
        row.appendChild(cell1);
        row.appendChild(cell2);
        row.appendChild(cell3);
        row.appendChild(cell4);
        row.appendChild(cell5);
        table.appendChild(row);
    }
    root.appendChild(table);
}

function makeProgressBar(root: HTMLElement, step: number, max_steps: number, time?: number) {
    while (root.firstChild) {
        root.removeChild(root.firstChild);
    }

    const wrapper = document.createElement("div");

    const progressBar = document.createElement("progress");
    progressBar.value = step;
    progressBar.max = max_steps;
    progressBar.classList.add("progress-bar");

    {
        const span = document.createElement("span");
        span.textContent = step + " / " + max_steps + "(" + Math.floor(100 * step / max_steps) + "%)";
        wrapper.appendChild(span);
    }

    if (time) {
        {
            const span = document.createElement("span");
            span.textContent = formatDuration(time);
            wrapper.appendChild(span);
        }
        {
            const span = document.createElement("span");
            const eta = time / step * max_steps - time;
            span.textContent = "ETA: " + formatDuration(eta);
            wrapper.appendChild(span);
        }
    }

    wrapper.appendChild(progressBar);
    root.appendChild(wrapper);
}

function updateJobsComponent(json: APIResponse.JSON, root: HTMLElement) {
    while (root.firstChild) {
        root.removeChild(root.firstChild);
    }

    const heading = document.createElement("h2");
    heading.textContent = "Jobs";
    root.appendChild(heading)

    const maxqueue = document.createElement("input");
    maxqueue.type = "number";
    maxqueue.value = String(json.max_queue);
    maxqueue.addEventListener("input", (e) => {
        fetch("/api/max_queue", {
            headers: [["Content-Type", "application/json"]],
            method: "POST",
            body: JSON.stringify({ "value": +maxqueue.value })
        });
    });


    const split = document.createElement("div");
    split.classList.add("split");
    root.appendChild(split);

    addRunningJobs("Running", json.job_threads, split);
    addQueuedJobs("Queued", json.n_queued_jobs, json.queued_jobs, json.queue_summary, split);
    addQueuedJobs("Redo-Queued", json.n_redo_queued_jobs, json.redo_queued_jobs, json.redo_queue_summary, split);

    function addRunningJobs(heading_str: string, threads: APIResponse.JobThread[], split: HTMLElement) {
        const tilesWrapper = document.createElement("div");
        tilesWrapper.classList.add("tile-view")
        split.appendChild(tilesWrapper);

        const heading = document.createElement("h3");
        heading.textContent = heading_str;
        tilesWrapper.appendChild(heading);

        const tiles = document.createElement("div");
        tiles.classList.add("tiles");
        tilesWrapper.appendChild(tiles);

        for (const thread of Object.values(threads)) {
            const workers = thread.worker_group.map((w) => w.client.id).join(",")

            const tile = document.createElement("div");
            tile.classList.add("tile");
            tile.textContent = thread.job.id + ": [" + workers + "] " + Math.floor(thread.runtime) + "s";

            const tileBar = document.createElement("div");
            makeProgressBar(tileBar, thread.job.iteration, thread.job.n_steps, thread.runtime);
            tile.appendChild(tileBar);

            tiles.appendChild(tile);
        }
    }

    function addQueuedJobs(heading_str: string, n_total_jobs: number, jobs: APIResponse.Job[], queue_summary: APIResponse.QueueSummary, split: HTMLElement) {
        const tilesWrapper = document.createElement("div");
        tilesWrapper.classList.add("tile-view")
        split.appendChild(tilesWrapper);

        const heading = document.createElement("h3");
        heading.textContent = heading_str;
        tilesWrapper.appendChild(heading);

        const tiles = document.createElement("div");
        tiles.classList.add("tiles");
        tilesWrapper.appendChild(tiles);

        for (const job of Object.values(jobs)) {
            const required_clients = job.required_clients.join(",")

            const tile = document.createElement("div");
            tile.classList.add("tile");
            tile.textContent = job.id + ": [" + required_clients + "]";

            tiles.appendChild(tile);
        }

        if (jobs.length < n_total_jobs) {
            const tile = document.createElement("div");
            tile.classList.add("tile");
            tile.textContent = (n_total_jobs - jobs.length) + " more";
            tiles.appendChild(tile);
        }

        const summaryDiv = document.createElement("div");
        {
            const backendSummary = document.createElement("ul");
            for (const [backend, amount] of Object.entries(queue_summary.backends)) {
                const item = document.createElement("li");
                item.textContent = backend + ": " + amount;
                backendSummary.appendChild(item);
            }
            summaryDiv.appendChild(backendSummary);
        }
        {
            const combinationSummary = document.createElement("ul");
            for (const [combination, amount] of Object.entries(queue_summary.combinations)) {
                const item = document.createElement("li");
                item.textContent = combination + ": " + amount;
                combinationSummary.appendChild(item);
            }
            summaryDiv.appendChild(combinationSummary);
        }
        tilesWrapper.appendChild(summaryDiv);
    }
}

function updateWorkersComponent(json: APIResponse.JSON, root: HTMLElement) {
    while (root.firstChild) {
        root.removeChild(root.firstChild);
    }

    const heading = document.createElement("h2");
    heading.textContent = "Workers";
    root.appendChild(heading)

    const split = document.createElement("div");
    split.classList.add("split");
    root.appendChild(split);

    addTiles("Idle", "IDLE", json, split);
    addTiles("Busy", "BUSY", json, split);

    function addTiles(heading_str: string, type: string, json: APIResponse.JSON, split: HTMLElement) {
        const tilesWrapper = document.createElement("div");
        tilesWrapper.classList.add("tile-view")
        split.appendChild(tilesWrapper);

        const heading = document.createElement("h3");
        heading.textContent = heading_str;
        tilesWrapper.appendChild(heading);

        const tiles = document.createElement("div");
        tiles.classList.add("tiles");
        tilesWrapper.appendChild(tiles);

        for (const worker of Object.values(json.workers)) {
            if (worker.state !== type) {
                continue;
            }

            const tile = document.createElement("div");
            tile.classList.add("tile");
            tile.textContent = worker.id + ": " + worker.client.id + "(cap: " + worker.current_capacity + ")";

            const input = document.createElement("input");
            input.type = "number";
            input.value = String(worker.max_jobs);
            input.addEventListener("input", (e) => {
                fetch("/api/" + worker.id + "/max_jobs", {
                    headers: [["Content-Type", "application/json"]],
                    method: "POST",
                    body: JSON.stringify({ "value": +input.value })
                });
            });
            tile.append(input);

            tiles.appendChild(tile);
        }
    }
}

function formatDuration(seconds: number) {
    const days = Math.floor(seconds / (24 * 3600));
    seconds %= (24 * 3600);
    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;
    const minutes = Math.floor(seconds / 60);
    seconds %= 60;

    seconds = Math.floor(seconds);

    const parts = [];
    if (days) parts.push(`${days}d`);
    if (hours) parts.push(`${hours}h`);
    if (minutes) parts.push(`${minutes}min`);
    parts.push(`${seconds}sec`);

    return parts.join(" ");
}


async function update() {
    const data = await fetch("/api/data.json", {
        method: "GET"
    });
    const json: APIResponse.JSON = await data.json();

    if (json.progress.total_steps) {
        makeProgressBar($("total-progress"), json.progress.iteration, json.progress.total_steps, json.elapsed_time);
    }

    updateBackendComponent(json, $("backend-overview"));
    updateWorkersComponent(json, $("workers-overview"));
    updateJobsComponent(json, $("jobs-overview"));

    setTimeout(update, 10000);
}

function $<T>(s: string) {
    return document.getElementById(s) as T;
}

((async function main() {
    const wrapper = document.createElement("div");
    document.body.appendChild(wrapper)

    setTimeout(update, 0);
})());
